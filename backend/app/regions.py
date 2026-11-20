"""Region proposals and multi-vector retrieval; no detector is loaded by the web app."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import faiss


def expand(box, fraction):
    x1,y1,x2,y2=box
    dx=(x2-x1)*fraction; dy=(y2-y1)*fraction
    return [max(0.,x1-dx),max(0.,y1-dy),min(1.,x2+dx),min(1.,y2+dy)]


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def propose_regions(detections):
    clean=[]
    for d in detections:
        box=np.clip(d['box'],0,1).tolist()
        if len(box)!=4 or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
            continue
        item={**d,'box':box}
        if not any(x['label']==item['label'] and iou(x['box'],box)>.85 for x in clean):
            clean.append(item)
    people=[d for d in clean if 'person' in d['label']]
    objects=[d for d in clean if 'person' not in d['label']]
    regions=[]
    for person in people:
        near=[o for o in objects if iou(expand(person['box'],.2),o['box'])>0]
        near.sort(key=lambda o: sum(((o['box'][j]+o['box'][j+2]-person['box'][j]-person['box'][j+2])/2)**2 for j in (0,1)))
        for obj in near[:3]:
            a=person['box'];b=obj['box']
            regions.append({'box':expand([min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])],.1),'kind':'interaction_candidate','objects':[person,obj],'confidence':min(person['score'],obj['score'])})
    for d in people+objects:
        regions.append({'box':expand(d['box'],.1),'kind':'person' if d in people else 'object','objects':[d],'confidence':d['score']})
    priority={'interaction_candidate':0,'person':1,'object':2}
    regions.sort(key=lambda r:(priority[r['kind']],-r['confidence']))
    selected=[]
    for region in regions:
        if not any(iou(region['box'],r['box'])>.85 for r in selected):
            selected.append(region)
        if len(selected)==12: break
    return selected


def crop_region(image, box):
    w,h=image.size
    coords=(int(box[0]*w),int(box[1]*h),max(int(box[0]*w)+1,int(np.ceil(box[2]*w))),max(int(box[1]*h)+1,int(np.ceil(box[3]*h))))
    crop=image.crop(coords).convert('RGB'); side=max(crop.size)
    # CLIP mean RGB, padded instead of center-cropping away context.
    return ImageOps.pad(crop,(side,side),color=(123,117,104),centering=(.5,.5))


class RegionIndex:
    def __init__(self, vectors, records, ids, version):
        self.vectors=np.ascontiguousarray(vectors,dtype='float32')
        self.records=records; self.version=version
        if len(vectors)!=len(records): raise ValueError('Region/vector count mismatch')
        lookup={name:i for i,name in enumerate(ids)}
        for record in records:
            box = record['box']
            if len(box) != 4 or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                raise ValueError('Invalid region coordinates')
        self.owners=np.array([lookup[r['id']] for r in records],dtype=np.int64)

    @classmethod
    def load(cls, directory, ids, model, pretrained, dim):
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        # Applies to startup, hot reload and ZIP import: retired indexes must
        # never reintroduce YOLO boxes, vocabulary labels or static fixtures.
        if 'yolo' in str(manifest.get('detector', '')).lower():
            raise ValueError('YOLO region indexes have been retired; use whole-image CLIP until a replacement index is ready')
        if not manifest.get('complete') or manifest['ids']!=ids: raise ValueError('Incomplete or different image corpus')
        if manifest['model']!=model or manifest['pretrained']!=pretrained: raise ValueError('Region model mismatch')
        records=json.loads((directory/'regions.json').read_text(encoding='utf-8'))
        index=faiss.read_index(str(directory/'index.faiss'))
        if index.d!=dim: raise ValueError('Region dimension mismatch')
        vectors=index.reconstruct_n(0,index.ntotal)
        if not np.isfinite(vectors).all() or (len(vectors) and not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-3)):
            raise ValueError('Invalid region vectors')
        return cls(vectors,records,ids,manifest['version'])

    def rank(self, detail_vectors, global_vectors, ids, k, whole_vector=None, queries=None):
        # Every condition is reduced by owner before combining: never mix images.
        fallback=global_vectors @ detail_vectors.T
        scores=np.full_like(fallback,-np.inf)
        winners=np.full(fallback.shape,-1,dtype=np.int64)
        sims=self.vectors @ detail_vectors.T
        for j in range(detail_vectors.shape[0]):
            np.maximum.at(scores[:,j],self.owners,sims[:,j])
            for r,owner in enumerate(self.owners):
                if winners[owner,j]<0 and sims[r,j]==scores[owner,j]: winners[owner,j]=r
        missing=winners<0
        scores[missing]=fallback[missing]
        combined=scores.mean(axis=1)
        if whole_vector is not None: combined=.5*combined+.5*(global_vectors @ whole_vector.reshape(-1))
        order=np.argsort(-combined,kind='stable')[:max(0,min(k,len(ids)))]
        return [{'id':ids[i],'score':float(combined[i]),'matches':[{'query':queries[j] if queries else str(j),'score':float(scores[i,j]),'source':'region' if winners[i,j]>=0 else 'whole_image','box':self.records[winners[i,j]]['box'] if winners[i,j]>=0 else None,'kind':self.records[winners[i,j]]['kind'] if winners[i,j]>=0 else None,'objects':self.records[winners[i,j]].get('objects',[]) if winners[i,j]>=0 else []} for j in range(len(detail_vectors))]} for i in order]
