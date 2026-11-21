"""Portable RegionCLIP index, independent from the whole-image CLIP space."""
import hashlib,json
from pathlib import Path
import faiss
import numpy as np
from .regions import RegionIndex

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def has_unit_region_vectors(vectors):
    for offset in range(0, len(vectors), 2048):
        norms = np.linalg.norm(vectors[offset:offset + 2048], axis=1)
        if not np.allclose(norms, 1, atol=1e-3):
            return False
    return True


class RegionClipIndex(RegionIndex):
    def __init__(self, vectors, records, ids, version):
        super().__init__(vectors, records, ids, version)
        self.context_vectors = None
        self.rows_by_image = {image_id: [] for image_id in ids}
        for row, record in enumerate(records):
            if record['kind'] != 'interaction_candidate':
                self.rows_by_image[record['id']].append(row)

    @classmethod
    def load(cls,directory,ids):
        directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
        if 'yolo' in str(m.get('detector','')).lower():raise ValueError('YOLO indexes have been retired')
        if m.get('schema')!='regionclip-v1' or not m.get('complete') or m['ids']!=ids or m.get('dim')!=1024:
            raise ValueError('Incomplete, incompatible or pilot RegionCLIP index')
        required={'index.faiss','regions.json','text_encoder.onnx','text_parity.json','tokenizer_bpe.txt.gz'}
        if m.get('config', {}).get('context_expansion') == 0.2:
            required.add('context.npy')
        if set(m['files'])!=required:raise ValueError('Unexpected RegionCLIP file manifest')
        for name,checksum in m['files'].items():
            if sha256(directory/name)!=checksum:raise ValueError('Checksum mismatch: '+name)
        records=json.loads((directory/'regions.json').read_text())
        index = faiss.read_index(str(directory/'index.faiss'))
        if index.d != 1024 or index.ntotal != len(records) or len(records) != m['num_regions']:
            raise ValueError('Invalid RegionCLIP vectors')
        vectors = index.reconstruct_n(0, index.ntotal)
        del index
        if not has_unit_region_vectors(vectors):
            raise ValueError('Invalid RegionCLIP vectors')
        errors=[row['id'] for row in m.get('errors',[])]
        if not set(errors)<=set(ids) or len(set(errors))!=len(errors):raise ValueError('Invalid error image mapping')
        obj=cls(vectors,records,ids,m['version']);obj.manifest=m
        if m.get('config', {}).get('roi_source') == 'proposal_view':
            if m.get('missing_region_ids') != [image_id for image_id in ids if not obj.rows_by_image[image_id]]:
                raise ValueError('Invalid missing region IDs')
            for row in records:
                box, window = row['source_box'], row['source_window']
                if row['source_view'] not in range(6) or any(
                    len(coords) != 4 or not (0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1)
                    for coords in (box, window)
                ):
                    raise ValueError('Invalid RegionCLIP source view')
                projected = [window[i % 2] + value * (window[i % 2 + 2] - window[i % 2])
                             for i, value in enumerate(box)]
                if not np.allclose(projected, row['box'], rtol=0, atol=1e-6):
                    raise ValueError('Incorrect RegionCLIP box projection')
        if 'context.npy' in required:
            expanded = np.load(directory/'context.npy', mmap_mode='r', allow_pickle=False)
            if expanded.shape != vectors.shape or expanded.dtype != np.float32:
                raise ValueError('Invalid RegionCLIP context shape or dtype')
            if not has_unit_region_vectors(expanded):
                raise ValueError('Invalid RegionCLIP context vectors')
            obj.context_vectors = expanded
        import onnxruntime as ort
        from open_clip.tokenizer import SimpleTokenizer
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        obj.session=ort.InferenceSession(str(directory/'text_encoder.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        obj.tokenizer=SimpleTokenizer(bpe_path=str(directory/'tokenizer_bpe.txt.gz'))
        parity=json.loads((directory/'text_parity.json').read_text())
        actual=obj.encode(parity['texts']);expected=np.array(parity['vectors'],dtype='float32')
        if actual.shape!=expected.shape or not np.isfinite(expected).all() or np.min(np.sum(actual*expected,axis=1))<.9999:raise ValueError('RegionCLIP text encoder parity failed')
        return obj

    def encode(self,texts):
        tokens=self.tokenizer(texts).numpy().astype('int64')
        vectors=self.session.run(None,{'tokens':tokens})[0]
        if vectors.shape!=(len(texts),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid RegionCLIP text vectors')
        norms=np.linalg.norm(vectors,axis=1,keepdims=True)
        if np.any(norms<1e-8):raise ValueError('Zero RegionCLIP text vector')
        return np.ascontiguousarray(vectors/norms,dtype='float32')

    def shortlist_matches(self, candidates, texts, context_weight=0.0):
        """Score object proposals only, in the RegionCLIP embedding space."""
        matches = {candidate['id']: [] for candidate in candidates}
        if not texts:
            return matches
        rows = [row for image_id in matches for row in self.rows_by_image[image_id]]
        if context_weight and self.context_vectors is None:
            raise ValueError('This index does not include expanded region vectors')
        similarities = None
        if rows:
            detail = self.encode(texts).T
            similarities = self.vectors[rows] @ detail
            if context_weight:
                similarities = (1 - context_weight) * similarities + context_weight * (self.context_vectors[rows] @ detail)
        start = 0
        for image_id in matches:
            count = len(self.rows_by_image[image_id])
            if not count:
                matches[image_id] = [{'query': text, 'score': None, 'box': None} for text in texts]
                continue
            scores = similarities[start:start + count]
            for j, best in enumerate(scores.argmax(axis=0)):
                region_row = int(rows[start + best])
                record = self.records[region_row]
                matches[image_id].append({'query': texts[j], 'score': float(scores[best, j]),
                                          'box': record['box'], 'kind': record['kind'], 'region_id': f'{self.version}:{region_row}',
                                          **{key: record[key] for key in ('source_view', 'source_window') if key in record}})
            start += count
        return matches

    def rank_shortlist(self, ranking, texts, candidate_k, threshold, weight, context_weight=0.0):
        """Rerank a prefix of the CLIP ranking and append its untouched tail."""
        candidates = ranking[:candidate_k]
        matches = (self.shortlist_matches(candidates, texts, context_weight) if context_weight
                   else self.shortlist_matches(candidates, texts))
        reranked = []
        for rank, image in enumerate(candidates, 1):
            evidence = []
            for match in matches[image['id']]:
                supported = match['score'] is not None and match['score'] >= threshold
                evidence.append({**match, 'supported': supported})
            coverage = sum(match['supported'] for match in evidence) / len(texts) if texts else 0.0
            reranked.append({**image, 'score': 1.0 / (60 + rank) + weight * coverage / 60,
                             'clip_score': image['score'], 'clip_rank': rank, 'matches': evidence,
                             'region_evaluated': bool(texts)})
        reranked.sort(key=lambda item: -item['score'])
        reranked.extend({**image, 'score': 1.0 / (60 + rank), 'clip_score': image['score'],
                         'clip_rank': rank, 'matches': [], 'region_evaluated': False}
                        for rank, image in enumerate(ranking[candidate_k:], candidate_k + 1))
        return reranked

    def rank_fused(self,texts,global_scores,ids,k,use_global=True):
        detail=self.encode(texts);scores=np.full((len(ids),len(texts)),-np.inf,dtype='float32')
        sims=self.vectors@detail.T
        winners=np.full(scores.shape,-1,dtype='int64')
        for j in range(len(texts)):
            np.maximum.at(scores[:,j],self.owners,sims[:,j])
            for r,owner in enumerate(self.owners):
                if winners[owner,j]<0 and sims[r,j]==scores[owner,j]:winners[owner,j]=r
        has=np.isfinite(scores).all(axis=1)
        detail_score=scores.mean(axis=1)
        def ranks(values,eligible):
            order=np.flatnonzero(eligible);order=order[np.argsort(-values[order],kind='stable')]
            result=np.zeros(len(ids),dtype=int);result[order]=np.arange(1,len(order)+1);return result
        gr=ranks(global_scores,np.ones(len(ids),bool));rr=ranks(detail_score,has)
        fused=np.zeros(len(ids));fused[has]=(.5/(60+gr[has]) if use_global else 0)+(.5 if use_global else 1)/(60+rr[has])
        # Missing regions use only the whole-image branch, not a foreign vector.
        fused[~has]=.5/(60+gr[~has])
        results=[]
        for i in np.argsort(-fused,kind='stable')[:max(0,min(k,len(ids)))]:
            matches=[]
            for j,text in enumerate(texts):
                r=self.records[winners[i,j]] if has[i] else None
                matches.append({'query':text,'score':float(scores[i,j]) if r else float(global_scores[i]),'source':'region' if r else 'whole_image','box':r['box'] if r else None,'kind':r['kind'] if r else None,'region_id':f'{self.version}:{int(winners[i,j])}' if r else None,'objects':r.get('objects',[]) if r else [],'model':'RegionCLIP-RN50' if r else 'CLIP-ViT-B-32'})
            results.append({'id':ids[i],'score':float(fused[i]),'matches':matches,'ranking':{'method':'rrf','clip_rank':int(gr[i]),'regionclip_rank':int(rr[i]) if has[i] else None,'clip_score':float(global_scores[i]),'regionclip_score':float(detail_score[i]) if has[i] else None}})
        return results
