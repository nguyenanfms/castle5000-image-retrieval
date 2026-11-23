(() => {
  "use strict";

  let activeRetrievalTurn = null;
  let qaConversationId = crypto.randomUUID();
  const $ = (id) => document.getElementById(id);
  const searchV2 = location.pathname === "/search-v2/";

  const loginView = $("login-view");
  const searchView = $("search-view");
  const statusbar = $("statusbar");
  const loginForm = $("login-form");
  const loginError = $("login-error");
  const searchForm = $("search-form");
  const queryInput = $(searchV2 ? "query-v2" : "query-input");
  const searchMeta = $("search-meta");
  const grid = $("results-grid");
  const userName = $("user-name");
  const logoutBtn = $("logout-btn");

  const tabSearchBtn = $("tab-search-btn");
  const tabRetrievalQaBtn = $("tab-retrieval-qa-btn");
  const retrievalQaPanel = $("retrieval-qa-panel");
  const retrievalQaForm = $("retrieval-qa-form");
  const retrievalQaQuestion = $("retrieval-qa-question");
  const retrievalQaQuery = $("retrieval-qa-query");
  const retrievalQaLang = $("retrieval-qa-lang");
  const retrievalQaSubmit = $("retrieval-qa-submit");
  const qaChatHistory = $("qa-chat-history");
  const qaChatEmpty = $("qa-chat-empty");
  const qaContextBar = $("qa-context-bar");
  const qaContextLabel = $("qa-context-label");
  const qaViewCurrentImgBtn = $("qa-view-current-img-btn");
  const qaClearImgBtn = $("qa-clear-img-btn");
  const qaHintRow = $("qa-hint-row");

  const lightbox = $("lightbox");
  const lightboxImg = $("lightbox-img");
  const lightboxId = $("lightbox-id");
  const submitBtn = $("submit-btn");
  const closeLightboxBtn = $("close-lightbox-btn");
  const submitResult = $("submit-result");

  const lightboxQaSection = $("lightbox-qa-section");
  const lightboxOpenChatBtn = $("lightbox-open-chat-btn");
  const lightboxQaForm = $("lightbox-qa-form");
  const lightboxQaInput = $("lightbox-qa-input");
  const lightboxQaLang = $("lightbox-qa-lang");
  const lightboxQaBtn = $("lightbox-qa-btn");
  const lightboxQaResult = $("lightbox-qa-result");
  const lightboxQaStatus = $("lightbox-qa-status");
  const lightboxQaMeta = $("lightbox-qa-meta");
  const lightboxQaAnswer = $("lightbox-qa-answer");
  const lightboxQaEvidence = $("lightbox-qa-evidence");

  const confirmDialog = $("confirm-dialog");
  const confirmYes = $("confirm-yes");
  const confirmNo = $("confirm-no");

  const regionControls = $("search-v2-controls");
  const planEditor = $("search-plan-editor");
  const planStatus = $("search-plan-status");
  const planToggle = $("search-manual");
  const searchLanguage = $("search-language");
  const searchLimit = $("search-limit");
  let searchInProgress = false;
  let planKey = null;
  let preparedPlan = null;
  let selectedWorkflowKeys = new Set();
  let workflowSelectionChanged = false;
  let workflowPhase = "idle";
  let workflowResponse = null;
  let workflowStep = null;
  let followWorkflow = true;
  let selectedSearchImage = null;
  if (searchV2) {
    document.body.classList.add("search-v2");
    $("search-version-link").hidden = true;
    $("query-input").hidden = $("query-input").disabled = true;
    queryInput.hidden = queryInput.disabled = false;
    searchLanguage.hidden = false;
    searchLimit.hidden = false;
    searchForm.querySelector('button[type="submit"]').textContent = "Tìm kiếm";
    $("inspector-close").hidden = false;
  }
  const TASK_POLL_MS = 5000;

  let currentImageId = null;
  let currentResults = [];
  let taskRunning = false;
  let selectedEvaluationId = "";
  let currentTaskType = null;
  const qaEvaluation = $("qa-evaluation-id");
  const qaEvaluationStatus = $("qa-evaluation-status");
  const qaSubmitText = $("qa-submit-text");
  const qaSubmitAnswer = $("qa-submit-answer");
  const qaSubmitStatus = $("qa-submit-status");
  const qaSubmitHistory = $("qa-submit-history");
  // Answers sent for the current QA task. Grading is manual, so this list is the only record the user sees.
  let qaSubmitted = { task: null, answers: [] };
  // KIS images sent for the current task, and whether one was judged correct.
  let kisSubmitted = { ids: new Set(), correct: false };

  function renderQaSubmitHistory() {
    qaSubmitHistory.replaceChildren(...qaSubmitted.answers.map((a) => {
      const li = document.createElement("li");
      const time = document.createElement("time");
      time.textContent = a.time;
      li.append(time, ` ${a.text}`);
      return li;
    }));
  }

  function setQaTask(key) {
    if (qaSubmitted.task === key) return;
    qaSubmitted = { task: key, answers: [] };
    kisSubmitted = { ids: new Set(), correct: false };
    qaSubmitStatus.textContent = "";
    renderQaSubmitHistory();
  }

  // Same rule as is_qa_task in backend/app/dres_client.py (task type names are chosen by the organisers).
  function isQaTaskType(type) {
    const t = (type || "").toLowerCase();
    return t.includes("question") || t.includes("answer") || /(^|[^a-z])q\s*&?\s*a([^a-z]|$)/.test(t);
  }

  function updateQaSubmitAvailability() {
    qaSubmitAnswer.disabled = localQaMode || !selectedEvaluationId || !taskRunning || !isQaTaskType(currentTaskType) || !qaSubmitText.value.trim();
  }

  // An image cannot answer a QA task, so the KIS button stays off while one runs.
  function updateKisSubmitAvailability(task) {
    const qa = Boolean(task?.running) && isQaTaskType(task.task_type);
    submitBtn.disabled = !task?.running || !selectedEvaluationId || qa;
    submitBtn.title = qa ? "Task hiện tại là QA: nộp đáp án chữ ở khung “Nộp đáp án QA”." : "";
  }
  let localQaMode = false;

  // QA runtime state (idle -> validating -> waiting -> answered/insufficient/ambiguous/error)
  let qaStatusInfo = {
    enabled: false,
    ready: false,
    supported_modes: [],
    max_question_length: 1200,
  };
  let lightboxQaState = "idle";
  let lightboxQaSeq = 0;
  let lightboxQaAbort = null;

  let retrievalQaSeq = 0;
  let retrievalQaAbort = null;
  let activeQaImageId = null;

  async function api(path, options = {}) {
    const resp = await fetch(path, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      ...options,
    });
    let body = null;
    try {
      body = await resp.json();
    } catch {
      /* no body */
    }
    if (!resp.ok) {
      const message = body && body.message ? body.message : `HTTP ${resp.status}`;
      const code = body && body.code ? body.code : "error";
      const err = new Error(message);
      err.code = code;
      err.status = resp.status;
      err.retryAfter = resp.headers ? resp.headers.get("Retry-After") : null;
      throw err;
    }
    return body;
  }

  async function refreshQaStatus() {
    try {
      const status = await api("/api/qa/status");
      qaStatusInfo = status;
      const supportsSelected =
        status.enabled &&
        Array.isArray(status.supported_modes) &&
        status.supported_modes.includes("selected_image");
      const supportsRetrieval =
        status.enabled &&
        Array.isArray(status.supported_modes) &&
        status.supported_modes.includes("retrieval_qa");

      if (lightboxQaSection) {
        lightboxQaSection.hidden = searchV2 || !supportsSelected;
      }
      lightboxOpenChatBtn.hidden = !supportsSelected || !supportsRetrieval;
      if (tabRetrievalQaBtn) {
        tabRetrievalQaBtn.hidden = !supportsRetrieval;
      }
      if (status.max_question_length) {
        if (lightboxQaInput) lightboxQaInput.maxLength = status.max_question_length;
        if (retrievalQaQuestion) retrievalQaQuestion.maxLength = status.max_question_length;
        if (retrievalQaQuery) retrievalQaQuery.maxLength = status.max_question_length;
      }
    } catch {
      if (lightboxQaSection) lightboxQaSection.hidden = true;
      lightboxOpenChatBtn.hidden = true;
      if (tabRetrievalQaBtn) tabRetrievalQaBtn.hidden = true;
    }
  }

  function showLoggedIn(username, role, mode) {
    localQaMode = role === "LOCAL_QA";
    loginView.hidden = true;
    searchView.hidden = false;
    statusbar.hidden = false;
    userName.textContent = username;
    const modeBadge = $("session-mode");
    if (modeBadge) {
      modeBadge.hidden = !(mode === "demo" || mode === "real");
      modeBadge.className = `session-mode-badge ${mode || ""}`;
      modeBadge.textContent = mode === "demo" ? "Demo" : "Thi thật";
      modeBadge.title = mode === "demo" ? "DRES giả trong máy: không nộp bài thật" : "DRES của ban tổ chức";
    }
    logoutBtn.hidden = localQaMode;
    submitBtn.hidden = localQaMode;
    document.querySelectorAll(".competition-settings").forEach((el) => { el.hidden = localQaMode; });
    api("/api/region-status").catch(() => {});
    refreshQaStatus();
    $("password").value = "";
  }

  function showLoggedOut() {
    qaConversationId = crypto.randomUUID();
    abortLightboxQa();
    abortRetrievalQa();
    loginView.hidden = false;
    searchView.hidden = true;
    statusbar.hidden = true;
    // A demo session's evaluation must not carry over to a real login.
    selectedEvaluationId = "";
    if (qaEvaluation) qaEvaluation.value = "";
    loadLoginModes();
  }

  async function loadLoginModes() {
    try {
      const { modes } = await api("/api/login-modes");
      const demo = $("login-mode-demo");
      if (demo) demo.hidden = !modes.includes("demo");
      if (!modes.includes("demo")) {
        const real = loginForm.querySelector('input[name="login-mode"][value="real"]');
        if (real) real.checked = true;
      }
    } catch {
      /* Keep both choices; the server rejects a mode it does not offer. */
    }
  }

  loginForm.querySelectorAll('input[name="login-mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.checked && radio.value === "demo" && !$("username").value) {
        $("username").value = "student";
        $("password").value = "student";
      }
    });
  });

  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    loginError.hidden = true;
    const username = $("username").value;
    const password = $("password").value;
    const mode = loginForm.querySelector('input[name="login-mode"]:checked')?.value || "real";
    try {
      const data = await api("/api/login", {
        method: "POST",
        body: JSON.stringify({ username, password, mode }),
      });
      showLoggedIn(data.username, data.role, data.mode);
      pollTask();
    } catch (err) {
      loginError.textContent = err.message || "Đăng nhập thất bại";
      loginError.hidden = false;
    }
  });

  logoutBtn.addEventListener("click", async () => {
    try {
      await api("/api/logout", { method: "POST" });
    } catch {
      /* ignore */
    }
    showLoggedOut();
  });

  function switchModeTab(mode) {
    if (!tabSearchBtn || !tabRetrievalQaBtn || !retrievalQaPanel) return;
    const isRetrievalQa = mode === "retrieval_qa" && !tabRetrievalQaBtn.hidden;
    tabSearchBtn.classList.toggle("active", !isRetrievalQa);
    tabSearchBtn.setAttribute("aria-selected", String(!isRetrievalQa));
    tabRetrievalQaBtn.classList.toggle("active", isRetrievalQa);
    tabRetrievalQaBtn.setAttribute("aria-selected", String(isRetrievalQa));
    searchForm.hidden = isRetrievalQa;
    regionControls.hidden = !searchV2 || isRetrievalQa || workflowPhase === "idle";
    retrievalQaPanel.hidden = !isRetrievalQa;
    if (searchMeta) searchMeta.hidden = isRetrievalQa;
    $("search-results-heading").hidden = !searchV2 || isRetrievalQa || !currentResults.length;
    if (grid) grid.hidden = isRetrievalQa;
    if (isRetrievalQa && retrievalQaQuestion) {
      retrievalQaQuestion.focus();
    }
  }

  if (tabSearchBtn) {
    tabSearchBtn.addEventListener("click", () => switchModeTab("search"));
  }
  if (tabRetrievalQaBtn) {
    tabRetrievalQaBtn.addEventListener("click", () => switchModeTab("retrieval_qa"));
  }

  let showRejectedResults = false;
  function renderResults(results) {
    currentResults = results;
    const checked = searchV2 && results.some(row => row.conditions?.some(check => check.kind !== "present"));
    const unconfirmed = checked ? new Set(results.filter(row => row.verify?.match !== "yes" &&
      row.conditions?.some(check => check.status === "unknown")).map(row => row.id)) : new Set();
    const allUnconfirmed = unconfirmed.size > 0 && unconfirmed.size === results.length;
    $("search-results-title").textContent = "Ảnh tìm được";
    $("search-results-heading").hidden = !searchV2 || !results.length || grid.hidden;
    $("search-results-count").textContent = `${results.length} ảnh`;
    grid.innerHTML = "";
    let unconfirmedHeadingShown = false;
    for (const [index, r] of results.entries()) {
      if (false && !unconfirmedHeadingShown && !allUnconfirmed && unconfirmed.has(r.id)) {
        const heading = document.createElement("h3");
        heading.className = "search-unconfirmed-heading";
        heading.textContent = `Chưa xác minh · ${unconfirmed.size} ảnh`;
        grid.appendChild(heading);
        unconfirmedHeadingShown = true;
      }
      const tile = document.createElement(searchV2 ? "button" : "div");
      tile.className = "thumb";

      const img = document.createElement("img");
      img.src = r.thumb;
      img.loading = "lazy";
      img.dataset.id = r.id;
      const scoreText = !searchV2 && typeof r.score === "number" ? ` (score ${r.score.toFixed(3)})` : "";
      img.title = `${r.id}${scoreText}`;
      if (searchV2) {
        tile.style.setProperty("--item-delay", `${Math.min(index, 7) * 24}ms`);
        tile.type = "button";
        tile.dataset.id = r.id;
        tile.setAttribute("aria-pressed", String(selectedSearchImage === r.id));
        tile.addEventListener("click", () => openLightbox(r.id));
        img.alt = "";
        img.addEventListener("error", () => { img.src = "/images/" + encodeURIComponent(r.id); }, { once: true });
      } else {
        img.addEventListener("click", () => openLightbox(r.id));
      }

      const label = document.createElement("span");
      label.className = "thumb-id";
      label.textContent = searchV2 && r.final_rank ? `${r.final_rank}. ${r.id}` : r.id;

      tile.appendChild(img);
      tile.appendChild(label);
      if (searchV2 && r.verify) {
        const verdict = document.createElement("span");
        verdict.className = `verify-badge verify-${r.verify.match}`;
        verdict.textContent = VERIFY_LABEL[r.verify.match];
        verdict.title = r.verify.reason || "";
        tile.dataset.verify = r.verify.match;
        tile.appendChild(verdict);
        // Frames Gemini rejected are folded away so they do not crowd the grid.
        if (r.verify.match === "no") tile.hidden = !showRejectedResults;
      }
      grid.appendChild(tile);
    }
    const rejected = searchV2 ? results.filter(r => r.verify?.match === "no").length : 0;
    if (rejected) {
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "search-rejected-toggle";
      const label = () => showRejectedResults ? `Ẩn ${rejected} ảnh không khớp` : `Hiện ${rejected} ảnh Gemini đánh giá không khớp`;
      toggle.textContent = label();
      toggle.addEventListener("click", () => {
        showRejectedResults = !showRejectedResults;
        grid.querySelectorAll('.thumb[data-verify="no"]').forEach(tile => { tile.hidden = !showRejectedResults; });
        toggle.textContent = label();
      });
      grid.appendChild(toggle);
    }
    $("search-results-count").textContent = rejected && !showRejectedResults
      ? `${results.length - rejected} ảnh · ẩn ${rejected} không khớp` : `${results.length} ảnh`;
  }

  searchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;
    if (searchV2) {
      await runDemoSearch("plan");
      return;
    }
    const searchButton = searchForm.querySelector("button");
    searchButton.disabled = true;
    searchForm.setAttribute("aria-busy", "true");
    searchMeta.textContent = "Đang tìm...";
    try {
      const data = await api("/api/search", {
        method: "POST",
        body: JSON.stringify({ query, k: 50, auto_split: true }),
      });
      renderResults(data.results);
      searchMeta.textContent = `${data.results.length} kết quả, ${data.took_ms}ms. ${data.notice || ""}`;
    } catch (err) {
      if (err.status === 401) {
        showLoggedOut();
        return;
      }
      searchMeta.textContent = `Lỗi: ${err.message}`;
    } finally {
      searchButton.disabled = false;
      searchForm.removeAttribute("aria-busy");
    }
  });

  const conditionTypes = {
    absent: "Không có", exactly: "Đúng số lượng", at_least: "Ít nhất", at_most: "Nhiều nhất",
    left_of: "Bên trái", right_of: "Bên phải", above: "Phía trên", below: "Phía dưới", near: "Ở gần",
    action: "Hành động", other: "Điều kiện khác",
  };
  const spatialConditions = new Set(["left_of", "right_of", "above", "below", "near"]);
  const countConditions = new Set(["exactly", "at_least", "at_most"]);

  function addPhrase(value = "") {
    const phrases = $("search-phrases");
    if (phrases.children.length >= 5) return;
    const row = document.createElement("div");
    const input = document.createElement("input");
    input.value = value;
    input.maxLength = 1200;
    input.placeholder = "Ví dụ: faucet";
    input.setAttribute("aria-label", "Cụm đối tượng bằng tiếng Anh");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "search-quiet";
    remove.textContent = "Bỏ";
    remove.addEventListener("click", () => {
      row.remove();
      $("search-add-phrase").disabled = false;
      updatePlanLabel();
    });
    row.append(input, remove);
    phrases.append(row);
    $("search-add-phrase").disabled = phrases.children.length >= 5;
    updatePlanLabel();
    return input;
  }

  function addCondition({kind = "absent", subject = "", target = "", count = 1} = {}) {
    const container = $("search-conditions");
    if (container.children.length >= 5) return;
    const row = document.createElement("div");
    const type = document.createElement("select");
    type.setAttribute("aria-label", "Loại điều kiện");
    for (const [value, label] of Object.entries(conditionTypes)) type.add(new Option(label, value));
    type.value = kind;
    const item = document.createElement("input");
    item.value = subject;
    item.required = true;
    item.maxLength = 220;
    item.setAttribute("aria-label", "Nội dung điều kiện bằng tiếng Anh");
    const relative = document.createElement("input");
    relative.value = target ?? "";
    relative.maxLength = 220;
    relative.placeholder = "Đối tượng thứ hai bằng tiếng Anh";
    relative.setAttribute("aria-label", "Đối tượng thứ hai bằng tiếng Anh");
    const number = document.createElement("input");
    number.type = "number";
    number.min = 0;
    number.max = 99;
    number.value = count ?? 1;
    number.setAttribute("aria-label", "Số lượng");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "search-quiet";
    remove.textContent = "Bỏ";
    remove.addEventListener("click", () => { row.remove(); updatePlanLabel(); });
    function syncFields() {
      const related = spatialConditions.has(type.value);
      const counted = countConditions.has(type.value);
      relative.hidden = relative.disabled = !related;
      relative.required = related;
      number.hidden = number.disabled = !counted;
      number.required = counted;
      item.placeholder = ["action", "other"].includes(type.value) ? "Mô tả điều kiện bằng tiếng Anh" : "Ví dụ: person";
    }
    type.addEventListener("change", syncFields);
    syncFields();
    row.append(type, item, relative, number, remove);
    container.append(row);
    updatePlanLabel();
    return item;
  }

  function readConditions() {
    return [...$("search-conditions").children].map(({children: [type, item, relative, number]}) => ({
      kind: type.value, subject: item.value.trim(),
      ...(relative.disabled ? {} : {target: relative.value.trim()}),
      ...(number.disabled ? {} : {count: number.valueAsNumber}),
    }));
  }

  function updatePlanLabel() {
    $("search-add-condition").disabled = $("search-conditions").children.length >= 5;
    planToggle.textContent = !planEditor.hidden ? "Đóng chỉnh sửa" : preparedPlan ? "Sửa chi tiết" : "Nhập điều kiện thủ công";
  }

  function showPlanEditor(open) {
    planEditor.hidden = !open;
    planToggle.setAttribute("aria-expanded", String(open));
    if (open) followWorkflow = false;
    renderWorkflowPopup();
    updatePlanLabel();
  }

  function setPlan(query = "", details = [], conditions = []) {
    $("search-global").value = query;
    $("search-phrases").replaceChildren();
    $("search-conditions").replaceChildren();
    $("search-add-phrase").disabled = false;
    details.forEach(addPhrase);
    conditions.forEach(addCondition);
    updatePlanLabel();
  }

  function showPlanStatus(message = "") {
    planStatus.textContent = message;
    planStatus.hidden = !message;
  }

  function currentPlanKey() {
    return JSON.stringify([queryInput.value.trim(), searchLanguage.value]);
  }

  function renderWorkflow(phase, response = null) {
    workflowPhase = phase;
    workflowResponse = response;
    regionControls.hidden = searchForm.hidden;
    const queryState = {
      idle: ["idle", "Việt hoặc Anh"], planning: ["active", "Đang phân tích"],
      review: ["done", "Đã phân tích"],
      searching: ["done", "Đã chuẩn bị mô tả"], done: ["done", "Đã chuẩn bị mô tả"],
      "plan-error": ["idle", "Có thể nhập thủ công"], "search-error": ["done", "Đã chuẩn bị mô tả"],
    }[phase];
    const conditionState = preparedPlan ? [phase === "review" ? "current" : "done", workflowSelectionLabel()]
      : ["idle", "Chờ phân tích"];
    const resultsState = phase === "searching" ? ["active", "Đang tìm ảnh"]
      : phase === "done" ? ["done", "Đã tìm xong"]
      : ["idle", phase === "search-error" ? "Chưa có kết quả" : "Chờ tìm ảnh"];
    $("workflow-announcement").textContent = phase === "planning" ? queryState[1]
      : phase === "review" ? `Chọn từ khóa và điều kiện. ${conditionState[1]}.`
      : phase === "searching" ? resultsState[1] : phase === "done" ? `Tìm được ${response.results.length} ảnh` : "";
    $("search-flow").dataset.phase = phase;
    for (const [id, [state, text]] of Object.entries({query: queryState, conditions: conditionState, results: resultsState})) {
      const stage = $(`flow-${id}`);
      stage.dataset.state = state;
      stage.removeAttribute("aria-current");
      if (["active", "current"].includes(state) || (id === "query" && phase === "idle") || (id === "results" && phase === "done")) {
        stage.setAttribute("aria-current", "step");
      }
      $(`flow-${id}-state`).textContent = text;
    }
    renderWorkflowPopup();
    if (!followWorkflow) return;
    if (phase === "review") selectWorkflowStep("conditions");
    else selectWorkflowStep(null);
  }

  function selectWorkflowStep(step) {
    const changed = workflowStep !== step;
    workflowStep = step;
    renderWorkflowPopup();
    if (step && changed && !$("workflow-popup").hidden && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const body = $("workflow-popup-body");
      body.getAnimations().forEach((animation) => animation.cancel());
      body.animate([{transform: "translateY(-5px)", opacity: .25}, {transform: "translateY(0)", opacity: 1}],
        {duration: 240, easing: "cubic-bezier(.2,.7,.2,1)"});
    }
  }

  function renderWorkflowPopup() {
    const popup = $("workflow-popup");
    popup.hidden = !workflowStep || !planEditor.hidden;
    planToggle.hidden = !popup.hidden && workflowStep === "conditions";
    for (const node of $("search-v2-controls").querySelectorAll(".flow-node")) {
      node.setAttribute("aria-expanded", String(!popup.hidden && node.dataset.flowStep === workflowStep));
    }
    if (popup.hidden) return;
    popup.dataset.step = workflowStep;
    const result = workflowResponse;
    const panels = {
      query: ["Câu truy vấn", workflowPhase === "planning" ? "Đang chuyển câu của bạn thành mô tả tìm ảnh."
        : queryInput.value.trim() || "Nhập mô tả ảnh bằng tiếng Việt hoặc tiếng Anh.", ["query"]],
      conditions: [preparedPlan ? "Chọn mục đối chiếu" : "Từ khóa & điều kiện", preparedPlan ? (workflowChecks(preparedPlan).length
        ? (workflowSelectionChanged ? "Bấm Tìm lại để áp dụng lựa chọn mới." : "")
        : "Lượt này chỉ có mô tả toàn cảnh.")
        : "Các mục sẽ hiện sau khi phân tích câu truy vấn.", preparedPlan && (workflowPhase === "review" || workflowSelectionChanged)
        ? ["search", "edit"] : ["edit"]],
      results: ["Kết quả tìm ảnh", result ? (result.results.length
        ? `${result.results.length} ảnh${result.mode === "clip_region" ? ` · Đã đối chiếu ${result.candidate_k} ảnh ứng viên` : " · Tìm theo mô tả toàn cảnh"}`
        : result.notice || "Không tìm thấy ảnh phù hợp.") : workflowPhase === "searching" ? "Đang tìm và đối chiếu các mục đã chọn."
        : workflowPhase === "search-error" ? "Chưa tìm được ảnh. Bạn có thể thử lại."
        : "Bấm Tìm kiếm để lấy ảnh.", result?.results.length ? ["results"] : (workflowPhase === "search-error" && preparedPlan ? ["search"] : [])],
    };
    const [title, message, actions] = panels[workflowStep];
    $("workflow-popup-title").textContent = title;
    $("workflow-popup-message").textContent = message;
    $("workflow-popup-message").hidden = !message;
    $("workflow-plan").hidden = workflowStep !== "conditions" || !preparedPlan;
    for (const toggle of $("workflow-plan").querySelectorAll('input[type="checkbox"]')) toggle.disabled = searchInProgress;
    for (const button of $("workflow-popup-actions").querySelectorAll("button")) {
      button.hidden = !actions.includes(button.dataset.workflowAction);
      button.disabled = searchInProgress && button.dataset.workflowAction !== "query";
    }
  }

  function workflowChecks(plan) {
    return plan ? [...plan.detailQueries.map((subject) => ({kind: "present", subject})), ...plan.conditions] : [];
  }

  function workflowKey({kind, subject, target, count}) {
    return JSON.stringify([kind, subject, target ?? null, count ?? null]);
  }

  function workflowSelectionLabel() {
    const total = $("workflow-plan").querySelectorAll('input[type="checkbox"]').length;
    return total ? `${selectedWorkflowKeys.size}/${total} mục đã chọn` : "Mô tả toàn cảnh";
  }

  function showWorkflowPlan(plan, keepSelection = false) {
    const excluded = new Set(keepSelection ? workflowChecks(preparedPlan).map(workflowKey)
      .filter((key) => !selectedWorkflowKeys.has(key)) : []);
    preparedPlan = plan;
    selectedWorkflowKeys = new Set();
    workflowSelectionChanged = false;
    $("workflow-caption").textContent = plan?.query || "";
    const objects = $("workflow-objects");
    const conditions = $("workflow-conditions");
    objects.replaceChildren();
    conditions.replaceChildren();
    let bubbleIndex = 0;
    for (const [key, check] of new Map(workflowChecks(plan).map((check) => [workflowKey(check), check]))) {
      const chip = document.createElement("li");
      chip.dataset.kind = check.kind;
      chip.style.setProperty("--bubble-delay", `${bubbleIndex * 58}ms`);
      bubbleIndex += 1;
      const label = document.createElement("label");
      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.value = key;
      toggle.checked = !excluded.has(key);
      if (toggle.checked) selectedWorkflowKeys.add(key);
      const text = document.createElement("span");
      text.textContent = conditionLabel(check);
      label.append(toggle, text);
      chip.append(label);
      (check.kind === "present" ? objects : conditions).append(chip);
    }
    $("workflow-objects-group").hidden = !objects.children.length;
    $("workflow-conditions-group").hidden = !conditions.children.length;
    renderWorkflowPopup();
    updatePlanLabel();
  }

  async function runDemoSearch(action) {
    if (searchInProgress || (action === "selected" && !preparedPlan)) return;
    searchInProgress = true;
    searchForm.setAttribute("aria-busy", "true");
    const button = searchForm.querySelector("button");
    const apply = $("search-apply-plan");
    button.disabled = apply.disabled = true;
    planToggle.disabled = true;
    searchLimit.disabled = true;
    try {
      if (action === "plan") {
        const key = currentPlanKey();
        const [query, language] = JSON.parse(key);
        showPlanStatus();
        showPlanEditor(false);
        followWorkflow = true;
        showWorkflowPlan(null);
        renderWorkflow("planning");
        searchMeta.textContent = "";
        renderResults([]);
        startThinking(true);
        setThinking("plan", "active", "Gemini tách cảnh, cụm đối tượng và điều kiện…");
        try {
          const plan = await api("/api/search/plan", {
            method: "POST", body: JSON.stringify({ query, language }),
          });
          setThinking("plan", "done", plan.global_query_en, {
            chips: [...plan.detail_queries_en.map((phrase) => [phrase, "object"]),
                    ...(plan.conditions || []).map((condition) => [conditionLabel(condition), "condition"])],
          });
          if (currentPlanKey() !== key) {
            renderWorkflow("idle");
            showPlanStatus("Bấm Tìm kiếm để dùng câu vừa sửa.");
            return;
          }
          setPlan(plan.global_query_en, plan.detail_queries_en, plan.conditions);
          planKey = key;
          showPlanStatus();
        } catch (err) {
          if (err.status === 401) throw err;
          // The whole-image model reads Vietnamese too: search with the typed sentence instead of stopping.
          setThinking("plan", "error", `${err.message} Tìm thẳng bằng câu gốc, không tách cụm.`);
          setPlan(query);
          planKey = key;
        }
      }
      followWorkflow = true;
      showPlanStatus();
      if (action !== "selected") {
        const query = $("search-global").value.trim();
        if (!query) {
          showPlanEditor(true);
          $("search-global").focus();
          return;
        }
        const detailQueries = [...$("search-phrases").querySelectorAll("input")]
          .map((input) => input.value.trim()).filter(Boolean);
        showPlanEditor(false);
        showWorkflowPlan({query, detailQueries, conditions: readConditions()}, action === "edited");
        searchMeta.textContent = "";
        renderResults([]);
        if (action === "edited") {
          renderWorkflow("review");
          return;
        }
      }
      const {query} = preparedPlan;
      const selected = workflowChecks(preparedPlan).filter((check) => selectedWorkflowKeys.has(workflowKey(check)));
      const detailQueries = [...new Set(selected.filter((check) => check.kind === "present").map((check) => check.subject))];
      const conditions = selected.filter((check) => check.kind !== "present");
      showPlanEditor(false);
      workflowSelectionChanged = false;
      renderWorkflow("searching");
      searchMeta.textContent = "";
      renderResults([]);
      if (!thinking || action !== "plan") startThinking(false);
      setThinking("search", "active", "Tìm theo toàn ảnh, rồi đối chiếu từng cụm theo vùng…");
      const response = await api("/api/search", {
        method: "POST",
        body: JSON.stringify({ query, detail_queries: detailQueries, conditions, mode: "clip_region", k: Number(searchLimit.value) }),
      });
      renderResults(response.results);
      renderWorkflow("done", response);
      searchMeta.textContent = response.notice || (response.results.length ? "" : "Không tìm thấy ảnh phù hợp.");
      setThinking("search", "done", response.mode === "clip_region"
        ? `${response.results.length} ảnh · đối chiếu vùng trên ${response.candidate_k} ứng viên`
        : `${response.results.length} ảnh · theo mô tả toàn cảnh`);
      const [filterState, filterDetail, filterChips] = summarizeFilter({ conditions }, response.results);
      setThinking("filter", filterState, filterDetail, { chips: filterChips });
      verifyTopResults({ query, detailQueries, conditions }, response.results);
    } catch (err) {
      if (err.status === 401) showLoggedOut();
      else {
        renderWorkflow("search-error");
        searchMeta.textContent = err.message;
        if (thinking?.steps.search.state === "active") {
          setThinking("search", "error", err.message);
          for (const key of ["filter", "verify", "zoom"]) setThinking(key, "skip", "");
        }
      }
    } finally {
      searchInProgress = false;
      searchForm.removeAttribute("aria-busy");
      button.disabled = apply.disabled = false;
      planToggle.disabled = false;
      searchLimit.disabled = false;
      renderWorkflowPopup();
      if (action === "edited" && planEditor.hidden) $("flow-conditions").querySelector("button").focus();
      if (action === "selected" && workflowStep === "results") $("flow-results").querySelector("button").focus();
    }
  }

  if (searchV2) {
    searchLimit.addEventListener("change", () => {
      if (workflowPhase === "done" && !workflowSelectionChanged && currentPlanKey() === planKey) {
        runDemoSearch("selected");
      }
    });
    planToggle.addEventListener("click", () => {
      if (!planEditor.hidden) {
        showPlanEditor(false);
        return;
      }
      if (currentPlanKey() !== planKey && !preparedPlan) {
        setPlan(queryInput.value.trim());
      }
      showPlanEditor(true);
      $("search-global").focus();
    });
    queryInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        searchForm.requestSubmit();
      }
    });
    $("search-add-phrase").addEventListener("click", () => addPhrase()?.focus());
    $("search-add-condition").addEventListener("click", () => addCondition()?.focus());
    $("search-thinking").addEventListener("click", (event) => {
      const node = event.target.closest(".flow-node");
      if (!node) return;
      const next = node.getAttribute("aria-expanded") === "true" ? null : node.dataset.flowStep;
      followWorkflow = false;
      showPlanEditor(false);
      selectWorkflowStep(next);
      renderThinking();
    });
    $("search-flow").addEventListener("click", (event) => {
      const node = event.target.closest(".flow-node");
      if (!node) return;
      const next = node.getAttribute("aria-expanded") === "true" ? null : node.dataset.flowStep;
      followWorkflow = false;
      showPlanEditor(false);
      selectWorkflowStep(next);
    });
    regionControls.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || $("workflow-popup").hidden) return;
      const activeNode = $("search-flow").querySelector('[aria-expanded="true"]');
      followWorkflow = false;
      selectWorkflowStep(null);
      activeNode?.focus();
    });
    $("workflow-popup-actions").addEventListener("click", (event) => {
      const action = event.target.closest("button")?.dataset.workflowAction;
      if (action === "query") queryInput.focus();
      if (action === "edit") planToggle.click();
      if (action === "search") runDemoSearch("selected");
      if (action === "results") grid.querySelector(".thumb")?.click();
    });
    $("workflow-plan").addEventListener("change", (event) => {
      const toggle = event.target;
      if (!toggle.matches('input[type="checkbox"]')) return;
      if (toggle.checked) selectedWorkflowKeys.add(toggle.value);
      else selectedWorkflowKeys.delete(toggle.value);
      if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
        const chip = toggle.closest("label");
        chip.getAnimations().forEach((animation) => { if (animation.id === "bubble-selection") animation.cancel(); });
        chip.animate([{transform: "scale(.94)"}, {transform: "scale(1.04)", offset: .55}, {transform: "scale(1)"}],
          {duration: 220, easing: "ease-out", id: "bubble-selection"});
      }
      if (workflowPhase === "done") workflowSelectionChanged = true;
      $("flow-conditions-state").textContent = workflowSelectionLabel();
      renderWorkflowPopup();
    });
    $("search-plan-form").addEventListener("submit", (event) => {
      event.preventDefault();
      planKey = currentPlanKey();
      runDemoSearch("edited");
    });
  }

  const VERIFY_LABEL = { yes: "Khớp", partial: "Một phần", no: "Không khớp" };
  const VERIFY_STATUS = { met: "Đạt", violated: "Không đạt", unclear: "Chưa rõ" };
  let verifySeq = 0;
  let lastVerification = null;
  const verifyToggle = $("search-verify-toggle");
  // ---- Search "thinking": one live line per stage, with timing and a short result. ----
  const THINKING_STEPS = [
    ["plan", "Hiểu câu", "plan"], ["search", "Tìm", "search"], ["filter", "Lọc điều kiện", "filter"],
    ["verify", "Gemini kiểm", "verify"], ["zoom", "Phóng to", "zoom"],
  ];
  let thinking = null;
  let thinkingTimer = null;

  function startThinking(planned) {
    thinking = { started: performance.now(), steps: {} };
    for (const [key] of THINKING_STEPS) thinking.steps[key] = { state: "todo", detail: "" };
    if (!planned) setThinking("plan", "skip", "Dùng mô tả đã có");
    clearInterval(thinkingTimer);
    thinkingTimer = setInterval(renderThinking, 500);
    renderThinking();
  }

  function setThinking(key, state, detail = "", extra = {}) {
    if (!thinking) return;
    const step = thinking.steps[key];
    if (state === "active") step.since = performance.now();
    if (["done", "error", "warn"].includes(state) && step.since) step.ms = performance.now() - step.since;
    Object.assign(step, { state, detail, ...extra });
    if (THINKING_STEPS.every(([k]) => !["todo", "active"].includes(thinking.steps[k].state))) {
      clearInterval(thinkingTimer);
      thinking.finished = performance.now();
    }
    renderThinking();
  }

  function formatSeconds(ms) {
    return `${(ms / 1000).toLocaleString("vi-VN", { maximumFractionDigits: 1, minimumFractionDigits: ms < 10000 ? 1 : 0 })} giây`;
  }

  function renderThinking() {
    const section = $("search-thinking");
    if (!section) return;
    section.hidden = !thinking;
    if (!thinking) return;
    const total = (thinking.finished || performance.now()) - thinking.started;
    $("thinking-clock").textContent = thinking.finished ? `Xong sau ${formatSeconds(total)}` : formatSeconds(total);
    const list = $("thinking-steps");
    list.replaceChildren();
    for (const [key, label, icon] of THINKING_STEPS) {
      const step = thinking.steps[key];
      const item = document.createElement("li");
      item.dataset.state = step.state;
      item.dataset.icon = icon;
      const panelStep = { plan: "conditions", search: "results" }[key];
      const clickable = panelStep && step.state !== "todo" && !(key === "plan" && !preparedPlan);
      const head = document.createElement(clickable ? "button" : "div");
      head.className = "thinking-step-head" + (clickable ? " flow-node" : "");
      if (clickable) {
        head.type = "button";
        head.dataset.flowStep = panelStep;
        head.setAttribute("aria-controls", "workflow-popup");
        head.setAttribute("aria-expanded", String(!$("workflow-popup").hidden && workflowStep === panelStep));
        head.title = key === "plan" ? "Xem và chọn từ khóa, điều kiện" : "Xem tóm tắt kết quả";
      }
      const dot = document.createElement("span");
      dot.className = "thinking-dot";
      dot.setAttribute("aria-hidden", "true");
      const name = document.createElement("strong");
      name.textContent = label;
      const time = document.createElement("span");
      time.className = "thinking-time";
      time.textContent = step.state === "active" && step.since ? formatSeconds(performance.now() - step.since)
        : step.ms ? formatSeconds(step.ms) : "";
      head.append(dot, name, time);
      item.append(head);
      const detailText = step.detail || (step.state === "skip" ? "Bỏ qua" : step.state === "todo" ? "Chờ" : "");
      if (detailText) {
        const detail = document.createElement("p");
        detail.textContent = detailText;
        item.append(detail);
      }
      if (step.chips?.length) {
        const chips = document.createElement("div");
        chips.className = "thinking-chips";
        for (const [text, kind] of step.chips) {
          const chip = document.createElement("span");
          chip.className = `thinking-chip chip-${kind}`;
          chip.textContent = text;
          chips.append(chip);
        }
        item.append(chips);
      }
      if (step.action) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "search-quiet thinking-action";
        button.textContent = step.action.label;
        button.addEventListener("click", step.action.run);
        item.append(button);
      }
      list.append(item);
    }
  }

  function summarizeFilter(plan, results) {
    const conditions = plan.conditions || [];
    if (!conditions.length) return ["skip", "Không có điều kiện", []];
    const chips = conditions.map((condition) => {
      const key = `${condition.kind}|${condition.subject}|${condition.target || ""}|${condition.count ?? ""}`;
      let met = 0, unknown = 0;
      for (const row of results) {
        const check = row.conditions?.find((c) => `${c.kind}|${c.subject}|${c.target || ""}|${c.count ?? ""}` === key);
        if (check?.status === "met") met += 1;
        else unknown += 1;
      }
      return [`${conditionLabel(condition)}: ${met} đạt · ${unknown} chưa rõ`, met ? "ok" : "muted"];
    });
    return ["done", "Ảnh vi phạm đã bị loại khỏi kết quả", chips];
  }

  function summarizeZoom(verdicts) {
    const zooms = Object.values(verdicts).map((v) => v.zoom).filter(Boolean);
    if (!zooms.length) return ["skip", "Không có ảnh cần xác nhận", []];
    const count = (status) => zooms.filter((z) => z.status === status).length;
    const chips = [[`${count("met")} xác nhận`, "ok"], [`${count("violated")} bác bỏ`, "bad"], [`${count("unclear")} chưa rõ`, "muted"]]
      .filter(([text]) => !text.startsWith("0 "));
    return ["done", "Cắt vùng chủ thể ở độ phân giải gốc", chips];
  }

  try {
    if (verifyToggle && localStorage.getItem("castle_verify") === "off") verifyToggle.checked = false;
  } catch { /* storage unavailable */ }
  verifyToggle?.addEventListener("change", () => {
    try { localStorage.setItem("castle_verify", verifyToggle.checked ? "on" : "off"); } catch { /* ignore */ }
  });

  // Confirmed matches first; partial and unchecked keep their search order; rejected last.
  function orderByVerification(rows) {
    const rank = (row) => ({ yes: 0, no: 2 })[row.verify?.match] ?? 1;
    return rows.map((row, index) => ({ row, index }))
      .sort((a, b) => rank(a.row) - rank(b.row) || a.index - b.index)
      .map(({ row }) => row);
  }

  let lastVerifyPlan = null;
  async function verifyTopResults(plan, results, retry = false) {
    const seq = ++verifySeq;
    lastVerifyPlan = plan;
    if (!retry) lastVerification = null;
    if (!searchV2 || !verifyToggle?.checked || !results.length) {
      setThinking("verify", "skip", !results.length ? "Không có ảnh để kiểm" : "Đã tắt kiểm tra bằng Gemini");
      setThinking("zoom", "skip", "");
      return;
    }
    setThinking("verify", "active", retry ? "Kiểm lại các ảnh chưa kiểm…" : "Đọc các ảnh đầu, bỏ ảnh gần trùng…");
    setThinking("zoom", "todo", "");
    try {
      const data = await api("/api/search/verify", {
        method: "POST",
        body: JSON.stringify({ query: plan.query, detail_queries: plan.detailQueries, conditions: plan.conditions,
          image_ids: results.slice(0, 50).map((row) => row.id), language: "vi" }),
      });
      if (seq !== verifySeq) return;
      // A retry adds verdicts for the images that failed before; earlier verdicts stay.
      lastVerification = retry && lastVerification
        ? { ...data, verdicts: { ...lastVerification.verdicts, ...data.verdicts } } : data;
      const verdicts = lastVerification.verdicts;
      const rows = currentResults.map((row) => verdicts[row.id] ? { ...row, verify: verdicts[row.id] } : row);
      renderResults(orderByVerification(rows));
      const counts = { yes: 0, partial: 0, no: 0 };
      Object.values(verdicts).forEach((v) => { counts[v.match] += 1; });
      const checkedCount = Object.keys(verdicts).length;
      const chips = [[`${counts.yes} khớp`, "ok"], [`${counts.partial} một phần`, "warn"], [`${counts.no} không khớp`, "bad"]]
        .filter(([text]) => !text.startsWith("0 "));
      if (data.skipped_near_duplicates) chips.push([`bỏ ${data.skipped_near_duplicates} ảnh gần trùng`, "muted"]);
      // More checks only on request: each click costs one or two Gemini calls.
      const unchecked = currentResults.filter((row) => !verdicts[row.id]);
      const retryAction = unchecked.length ? {
        label: data.errors?.length ? "Kiểm lại ảnh chưa kiểm" : `Kiểm thêm ${Math.min(4, unchecked.length)} ảnh`,
        run: () => verifyTopResults(lastVerifyPlan, currentResults.filter((row) => !row.verify), true),
      } : null;
      setThinking("verify", data.errors?.length ? "warn" : "done",
        data.errors?.length ? `Đã kiểm ${checkedCount} ảnh; một nhóm lỗi: ${data.errors[0]}`
          : `Đã kiểm ${checkedCount} ảnh · khớp lên đầu, không khớp xuống cuối`,
        { chips, action: retryAction });
      const [zoomState, zoomDetail, zoomChips] = summarizeZoom(verdicts);
      setThinking("zoom", zoomState, zoomDetail, { chips: zoomChips });
    } catch (err) {
      if (seq !== verifySeq) return;
      if (err.status === 401) { showLoggedOut(); return; }
      setThinking("verify", "error", `Chưa kiểm được: ${err.message} Kết quả giữ thứ tự tìm kiếm.`, {
        action: { label: "Thử lại", run: () => verifyTopResults(lastVerifyPlan, currentResults) },
      });
      setThinking("zoom", "skip", "");
    }
  }

  function renderVerification(result) {
    const panel = $("lightbox-verify");
    panel.replaceChildren();
    const verdict = result?.verify;
    panel.hidden = !verdict;
    if (!verdict) return;
    const head = document.createElement("div");
    head.className = "verify-head";
    const title = document.createElement("h2");
    title.textContent = "Gemini kiểm tra";
    const badge = document.createElement("span");
    badge.className = `verify-badge verify-${verdict.match}`;
    badge.textContent = VERIFY_LABEL[verdict.match];
    head.append(title, badge);
    panel.append(head);
    if (verdict.reason) {
      const reason = document.createElement("p");
      reason.className = "verify-reason";
      reason.textContent = verdict.reason;
      panel.append(reason);
    }
    if (verdict.zoom) {
      const zoom = document.createElement("p");
      zoom.className = "verify-zoom";
      zoom.dataset.status = verdict.zoom.status;
      const label = { met: "Ảnh cận cảnh xác nhận", violated: "Ảnh cận cảnh không đúng", unclear: "Ảnh cận cảnh chưa rõ" };
      zoom.textContent = `${label[verdict.zoom.status]}${verdict.zoom.note ? `: ${verdict.zoom.note}` : ""}`;
      panel.append(zoom);
    }
    const texts = new Map((lastVerification?.requirements || []).map((r) => [r.id, r.text]));
    const list = document.createElement("ul");
    list.className = "verify-list";
    for (const check of verdict.requirements) {
      const item = document.createElement("li");
      item.dataset.status = check.status;
      const state = document.createElement("small");
      state.textContent = VERIFY_STATUS[check.status];
      const text = document.createElement("span");
      text.textContent = (texts.get(check.id) || `Yêu cầu ${check.id}`) + (check.note ? ` — ${check.note}` : "");
      item.append(state, text);
      list.append(item);
    }
    panel.append(list);
    if (verdict.boxes?.length) {
      const show = document.createElement("button");
      show.type = "button";
      show.className = "search-quiet";
      show.textContent = "Xem vùng Gemini chỉ ra";
      show.addEventListener("click", () => highlightEvidenceRegions(verdict.boxes));
      panel.append(show);
    }
  }

  // Neighbouring frames of the same hour, to confirm the moment before submitting.
  let neighborIds = [];
  async function renderNeighbors(id) {
    const section = $("lightbox-neighbors");
    const list = $("lightbox-neighbor-list");
    if (!searchV2 || (retrievalQaPanel && !retrievalQaPanel.hidden)) { section.hidden = true; return; }
    try {
      const data = await api(`/api/neighbors/${encodeURIComponent(id)}?radius=4`);
      if (currentImageId !== id) return;
      neighborIds = data.frames.map((frame) => frame.id);
      list.replaceChildren();
      for (const frame of data.frames) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "neighbor-frame" + (frame.id === id ? " is-current" : "");
        button.setAttribute("aria-label", `Mở khung ${frame.id}`);
        if (frame.id === id) button.setAttribute("aria-current", "true");
        const img = document.createElement("img");
        img.src = "/thumbs/" + encodeURIComponent(frame.id);
        img.alt = "";
        img.loading = "lazy";
        const caption = document.createElement("span");
        caption.textContent = frame.id.replace(/\.jpg$/i, "");
        button.append(img, caption);
        button.addEventListener("click", () => openLightbox(frame.id));
        list.append(button);
      }
      section.hidden = data.frames.length < 2;
      list.querySelector(".is-current")?.scrollIntoView({ block: "nearest", inline: "center" });
    } catch {
      section.hidden = true;
    }
  }

  function stepNeighbor(direction) {
    const index = neighborIds.indexOf(currentImageId);
    const next = neighborIds[index + direction];
    if (index >= 0 && next) openLightbox(next);
  }

  function getNaturalOverlaySize() {
    const w = lightboxImg.naturalWidth > 0 ? lightboxImg.naturalWidth : 1280;
    const h = lightboxImg.naturalHeight > 0 ? lightboxImg.naturalHeight : 720;
    const overlay = $("matched-overlay");
    if (overlay) {
      overlay.setAttribute("viewBox", `0 0 ${w} ${h}`);
    }
    return { w, h };
  }

  function abortLightboxQa() {
    lightboxQaSeq += 1;
    if (lightboxQaAbort) {
      lightboxQaAbort.abort();
      lightboxQaAbort = null;
    }
    lightboxQaState = "idle";
    if (lightboxQaBtn) {
      lightboxQaBtn.disabled = false;
    }
  }

  function resetLightboxQaPanel(prefillQuestion = null) {
    abortLightboxQa();
    if (lightboxQaResult) lightboxQaResult.hidden = true;
    if (lightboxQaStatus) {
      lightboxQaStatus.className = "qa-badge";
      lightboxQaStatus.textContent = "";
    }
    if (lightboxQaMeta) lightboxQaMeta.textContent = "";
    if (lightboxQaAnswer) lightboxQaAnswer.textContent = "";
    if (lightboxQaEvidence) lightboxQaEvidence.replaceChildren();
    if (prefillQuestion !== null && lightboxQaInput) {
      lightboxQaInput.value = prefillQuestion;
    }
  }

  function openLightbox(id, prefillQuestion = null, qaEvidence = null) {
    const imageChanged = currentImageId !== id;
    currentImageId = id;
    lightboxImg.src = `/images/${id}`;
    lightboxImg.alt = id;
    submitResult.textContent = "";
    lightboxId.textContent = id;
    if (imageChanged || prefillQuestion !== null) {
      resetLightboxQaPanel(prefillQuestion);
    }
    // QA evidence must never inherit conditions or highlights from a prior KIS search.
    const inQa = Boolean(qaEvidence) || Boolean(retrievalQaPanel && !retrievalQaPanel.hidden);
    const result = inQa ? null : currentResults.find((row) => row.id === id);
    if (searchV2) {
      if (!inQa) {
        selectedSearchImage = id;
        const cards = grid.querySelectorAll(".thumb");
        cards.forEach((card) => card.setAttribute("aria-pressed", String(card.dataset.id === id)));
      }
      renderDemoEvidence(result, qaEvidence);
      renderVerification(result);
      renderNeighbors(id);
      // One verdict per image: once Gemini has checked it, the region list is only a hint.
      const regionPanel = $("search-region-evidence");
      if (regionPanel?.dataset) regionPanel.dataset.secondary = String(Boolean(result?.verify));
      const regionHeading = regionPanel?.querySelector?.(".evidence-header h2");
      if (regionHeading) regionHeading.textContent = result?.verify ? "Vùng RegionCLIP gợi ý" : "Điều kiện trên ảnh";
    } else {
      renderMatchedRegions(result?.matches || []);
      lightboxImg.onload = () => {
        renderMatchedRegions(inQa ? [] : currentResults.find((row) => row.id === currentImageId)?.matches || []);
      };
    }
    lightbox.hidden = false;
    if (searchV2) $("inspector-close").focus({preventScroll: true});
  }

  const conditionStatus = {met: "Đạt", violated: "Vi phạm", unknown: "Chưa xác minh được"};
  const conditionReason = {
    localized: "Vùng khớp vượt ngưỡng.", weak_region: "Vùng khớp chưa đủ bằng chứng.",
    excluded_object_detected: "Có khung phát hiện đối tượng cần loại trừ.",
    absence_unconfirmed: "Chưa có bằng chứng đủ mạnh để xác nhận sự vắng mặt.",
    detector_unavailable: "Chưa kiểm tra: tìm ảnh không chạy bộ dò đối tượng cho điều kiện đếm hoặc loại trừ.",
    person_region_detected: "Chỉ mục vùng có vùng được nhận là người trong ảnh này.",
    minimum_observed: "Đủ số khung riêng biệt để đạt mức tối thiểu.",
    count_exceeded: "Số khung riêng biệt đã vượt giới hạn.",
    count_unconfirmed: "Có thể còn đối tượng bị khuất hoặc bị bỏ sót.",
    spatial_match: "Vị trí hai vùng trên ảnh phù hợp với điều kiện.",
    spatial_unconfirmed: "Chưa đủ bằng chứng về vị trí của cả hai đối tượng.",
    action_unconfirmed: "Các vùng đối tượng chỉ cho biết chúng xuất hiện. Hành động cần bằng chứng riêng.",
    undetermined: "Chưa có phép kiểm tra tự động cho điều kiện này.",
  };

  function conditionLabel(check) {
    const subject = objectLabel(check.subject);
    if (check.kind === "present") return subject;
    const label = conditionTypes[check.kind];
    if (check.kind === "absent") return `${label} ${subject}`;
    if (spatialConditions.has(check.kind)) return `${subject} ${label.toLowerCase()} ${objectLabel(check.target)}`;
    if (countConditions.has(check.kind)) {
      return `${check.kind === "exactly" ? "Đúng" : label} ${check.count} ${subject}`;
    }
    return `${label} · ${subject}`;
  }

  function objectLabel(object) {
    return ({person: "người", people: "người", sink: "bồn rửa", faucet: "vòi nước",
      refrigerator: "tủ lạnh", "refrigerator door": "cánh tủ lạnh", "kitchen island": "bàn đảo bếp",
      cup: "cốc", bowl: "tô", plate: "đĩa", bottle: "chai"})[object?.toLowerCase()] || object;
  }

  function renderDemoEvidence(result, initialQaEvidence = null) {
    const panel = $("search-region-evidence");
    const overlay = $("matched-overlay");
    const visual = $("lightbox-visual");
    $("matched-toggle").hidden = true;
    $("matched-description").hidden = true;
    panel.replaceChildren();
    panel.hidden = !result;
    overlay.replaceChildren();
    overlay.setAttribute("hidden", "");
    overlay.style.removeProperty("display");
    visual.style.removeProperty("--photo-ratio");
    let selected = null;
    let qaEvidence = initialQaEvidence && (initialQaEvidence.regions?.length || Array.isArray(initialQaEvidence.box))
      ? initialQaEvidence : null;
    function draw() {
      if (lightboxImg.naturalWidth && lightboxImg.naturalHeight) {
        visual.style.setProperty("--photo-ratio", lightboxImg.naturalWidth / lightboxImg.naturalHeight);
      }
      overlay.replaceChildren();
      overlay.setAttribute("hidden", "");
      if (qaEvidence && lightboxImg.complete && lightboxImg.naturalWidth) {
        if (qaEvidence.regions?.length) highlightEvidenceRegions(qaEvidence.regions);
        else if (Array.isArray(qaEvidence.box)) highlightEvidenceBox(qaEvidence.box, qaEvidence.image_id);
        return;
      }
      if (!selected?.evidence.length || !lightboxImg.complete || !lightboxImg.naturalWidth) return;
      const { w, h } = getNaturalOverlaySize();
      const stroke = {met: "#d6ff7f", violated: "#ff9d91", unknown: "#ffd692"}[selected.status];
      for (const { box: coordinates } of selected.evidence) {
        if (!coordinates) continue;
        const [x1, y1, x2, y2] = coordinates;
        const box = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        for (const [key, value] of Object.entries({
          x: x1 * w, y: y1 * h, width: (x2 - x1) * w, height: (y2 - y1) * h,
          fill: "none", stroke, "stroke-width": 3, "vector-effect": "non-scaling-stroke",
        })) box.setAttribute(key, value);
        if (selected.status === "unknown") box.setAttribute("stroke-dasharray", "6 4");
        overlay.append(box);
      }
      if (overlay.children.length) overlay.removeAttribute("hidden");
    }
    lightboxImg.onload = draw;
    draw();
    if (!result) return;
    const rank = document.createElement("p");
    rank.className = "region-rank";
    rank.textContent = `Hạng ${result.final_rank ?? result.clip_rank}`;
    const checks = result.conditions?.length ? result.conditions : result.matches.map((match) => ({
      kind: "present", subject: match.query, status: match.supported ? "met" : "unknown",
      reason: match.supported ? "localized" : "weak_region", evidence: match.box ? [match] : [],
    }));
    const heading = document.createElement("h2");
    heading.textContent = "Điều kiện trên ảnh";
    const header = document.createElement("div");
    header.className = "evidence-header";
    header.append(heading, rank);
    panel.append(header);
    if (!checks.length) {
      const empty = document.createElement("p");
      empty.className = "region-match-note";
      empty.textContent = "Ảnh được tìm theo mô tả toàn cảnh.";
      panel.append(empty);
      return;
    }
    const phrases = document.createElement("div");
    phrases.className = "region-phrases";
    const detail = document.createElement("p");
    detail.className = "region-match-note";
    detail.setAttribute("role", "status");
    for (const check of checks) {
      const button = document.createElement("button");
      button.type = "button";
      const badge = document.createElement("small");
      badge.dataset.status = check.status;
      badge.textContent = conditionStatus[check.status];
      const label = document.createElement("span");
      label.textContent = conditionLabel(check);
      button.append(label, badge);
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => {
        qaEvidence = null;
        selected = check;
        for (const chip of phrases.children) chip.setAttribute("aria-pressed", String(chip === button));
        const lowerBound = check.observed_count > 0 ? `Phát hiện ít nhất ${check.observed_count}` : "";
        detail.textContent = [conditionReason[check.reason], lowerBound].filter(Boolean).join(" · ");
        draw();
      });
      phrases.append(button);
    }
    panel.append(phrases, detail);
    const initialEvidence = qaEvidence;
    phrases.children[Math.max(0, checks.findIndex((check) => check.evidence.some((point) => point.box)))]?.click();
    qaEvidence = initialEvidence;
    draw();
  }

  function closeLightbox() {
    if (lightbox.hidden) return;
    abortLightboxQa();
    lightbox.hidden = true;
    currentImageId = null;
    if (searchV2 && selectedSearchImage) {
      grid.querySelector(`button[data-id="${CSS.escape(selectedSearchImage)}"]`)?.focus({preventScroll: true});
    }
  }
  $("inspector-close").addEventListener("click", closeLightbox);

  function renderMatchedRegions(matches) {
    const overlay = $("matched-overlay");
    const legend = $("matched-description");
    const button = $("matched-toggle");
    overlay.replaceChildren();
    legend.replaceChildren();
    overlay.setAttribute("hidden", "");
    overlay.style.display = "none";
    legend.hidden = true;
    button.setAttribute("aria-pressed", "false");
    button.textContent = "Xem vùng khớp truy vấn";
    const valid = matches.filter((match) => match.source === "region" && Array.isArray(match.box));
    button.hidden = valid.length === 0;
    const { w: natW, h: natH } = getNaturalOverlaySize();
    const colors = ["#4ade80", "#60a5fa", "#fbbf24", "#f472b6", "#c4b5fd"];
    const ns = "http://www.w3.org/2000/svg";
    const node = (name, attrs) => {
      const item = document.createElementNS(ns, name);
      for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, String(value));
      return item;
    };
    valid.forEach((match, i) => {
      const color = colors[i % colors.length];
      const [x1, y1, x2, y2] = match.box;
      const detected = (match.objects || [])
        .map((obj) => obj.label)
        .filter((name, index, list) => list.indexOf(name) === index)
        .join(" + ");
      const caption = `${i + 1}. ${detected || "Vùng khớp"}`;
      const width = Math.min(natW - 20, caption.length * 9 + 16);
      const lx = Math.max(0, Math.min(x1 * natW, natW - width));
      const ly = Math.max(0, y1 * natH - 25);
      const group = node("g", {});
      group.append(
        node("rect", {
          x: x1 * natW,
          y: y1 * natH,
          width: (x2 - x1) * natW,
          height: (y2 - y1) * natH,
          fill: "none",
          stroke: color,
          "stroke-width": 3,
        })
      );
      group.append(node("rect", { x: lx, y: ly, width, height: 24, fill: color, rx: 3 }));
      const text = node("text", {
        x: lx + 6,
        y: ly + 17,
        fill: "#10141b",
        "font-size": 16,
        "font-family": "sans-serif",
      });
      text.textContent = caption;
      group.append(text);
      overlay.append(group);
      const line = document.createElement("p");
      line.style.color = color;
      line.textContent = `${i + 1}. ${match.query} · điểm khớp ${match.score.toFixed(3)}`;
      legend.append(line);
    });
    if (valid.length) {
      const note = document.createElement("p");
      note.className = "match-note";
      note.textContent = "Khung là vùng phù hợp với mô tả; chưa xác nhận hành động xảy ra đúng.";
      legend.append(note);
    }
  }

  function highlightEvidenceBox(box, labelText) {
    highlightEvidenceRegions([{box, label: labelText}]);
  }

  function highlightEvidenceRegions(regions) {
    const overlay = $("matched-overlay");
    const { w: natW, h: natH } = getNaturalOverlaySize();
    const ns = "http://www.w3.org/2000/svg";
    overlay.replaceChildren();
    regions.forEach(({box, label}, index) => {
      if (!Array.isArray(box) || box.length !== 4) return;
      const [x1, y1, x2, y2] = box;
      const rect = document.createElementNS(ns, "rect");
      rect.setAttribute("x", String(x1 * natW));
      rect.setAttribute("y", String(y1 * natH));
      rect.setAttribute("width", String((x2 - x1) * natW));
      rect.setAttribute("height", String((y2 - y1) * natH));
      rect.setAttribute("fill", "none");
      rect.setAttribute("stroke", "#4f8cff");
      rect.setAttribute("stroke-width", "4");
      overlay.appendChild(rect);
      if (label || regions.length > 1) {
        const txt = document.createElementNS(ns, "text");
        txt.setAttribute("x", String(Math.max(4, x1 * natW + 6)));
        txt.setAttribute("y", String(Math.max(18, y1 * natH - 6)));
        txt.setAttribute("fill", "#4f8cff");
        txt.setAttribute("font-size", "16");
        txt.setAttribute("font-family", "sans-serif");
        txt.setAttribute("paint-order", "stroke");
        txt.setAttribute("stroke", "#10141b");
        txt.setAttribute("stroke-width", "3");
        txt.textContent = label || String(index + 1);
        overlay.appendChild(txt);
      }
    });
    overlay.removeAttribute("hidden");
    overlay.style.display = "block";
  }

  $("matched-toggle").onclick = () => {
    const showing = $("matched-toggle").getAttribute("aria-pressed") !== "true";
    $("matched-overlay").toggleAttribute("hidden", !showing);
    $("matched-overlay").style.display = showing ? "block" : "none";
    $("matched-description").hidden = !showing;
    $("matched-toggle").setAttribute("aria-pressed", String(showing));
    $("matched-toggle").textContent = showing ? "Ẩn vùng khớp" : "Xem vùng khớp truy vấn";
  };

  function statusLabel(status, lang = "vi") {
    const vi = {
      validating: "Đang kiểm tra…",
      waiting: "Đang trả lời…",
      answered: "Đã trả lời",
      insufficient_evidence: "Không đủ bằng chứng",
      ambiguous: "Câu hỏi mơ hồ",
      error: "Lỗi",
    };
    const en = {
      validating: "Validating…",
      waiting: "Answering…",
      answered: "Answered",
      insufficient_evidence: "Insufficient evidence",
      ambiguous: "Ambiguous",
      error: "Error",
    };
    const table = lang === "en" ? en : vi;
    return table[status] || status;
  }

  function formatQaErrorMessage(err) {
    if (err.status === 429 && err.retryAfter) {
      return `${err.message} (Thử lại sau ${err.retryAfter}s)`;
    }
    return err.message || "Không thể xử lý yêu cầu hỏi đáp.";
  }

  function scrollChatToBottom() {
    if (!qaChatHistory) return;
    qaChatHistory.scrollTop = qaChatHistory.scrollHeight;
  }

  function setActiveQaImage(newImageId) {
    activeQaImageId = newImageId ? String(newImageId).trim() : null;
    qaContextBar.hidden = !activeQaImageId;
    if (qaContextLabel) {
      qaContextLabel.textContent = activeQaImageId
        ? `Ảnh đang hỏi · ${activeQaImageId}`
        : "";
    }
    if (qaViewCurrentImgBtn) {
      qaViewCurrentImgBtn.hidden = !activeQaImageId;
    }
    if (qaClearImgBtn) {
      qaClearImgBtn.hidden = !activeQaImageId;
    }
    if (qaHintRow) {
      qaHintRow.hidden = Boolean(activeQaImageId);
    }
    qaChatEmpty.querySelectorAll("[data-qa-pinned]").forEach((button) => {
      button.textContent = activeQaImageId ? button.dataset.qaPinned : button.dataset.qaSuggestion;
    });
    if (!activeRetrievalTurn) renderRetrievalPanel(null);
  }

  if (qaViewCurrentImgBtn) {
    qaViewCurrentImgBtn.addEventListener("click", () => {
      if (activeQaImageId) {
        openLightbox(activeQaImageId);
      }
    });
  }

  if (qaClearImgBtn) {
    qaClearImgBtn.addEventListener("click", () => {
      setActiveQaImage(null);
      if (retrievalQaQuestion) retrievalQaQuestion.focus();
    });
  }

  if (lightboxOpenChatBtn) {
    lightboxOpenChatBtn.addEventListener("click", () => {
      const targetId = currentImageId;
      closeLightbox();
      if (tabRetrievalQaBtn && !tabRetrievalQaBtn.hidden) {
        switchModeTab("retrieval_qa");
      }
      if (targetId) {
        setActiveQaImage(targetId);
      }
    });
  }

  function createEvidenceActionBtn(ev, labelText = "Xem ảnh bằng chứng") {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "qa-evidence-btn";
    btn.textContent = labelText;
    btn.title = ev.image_id;
    btn.addEventListener("click", () => {
      openLightbox(ev.image_id, null, ev);
      if (!searchV2 && ev.regions?.length) {
        const showRegions = () => {
          if (currentImageId === ev.image_id) highlightEvidenceRegions(ev.regions);
        };
        if (lightboxImg.complete && lightboxImg.naturalWidth) showRegions();
        else lightboxImg.addEventListener("load", showRegions, { once: true });
      } else if (!searchV2 && Array.isArray(ev.box)) {
        highlightEvidenceBox(ev.box, ev.image_id);
      }
    });
    return btn;
  }

  function createCosineSummary(member, estimate = null, evidenceByImage = new Map()) {
    const line = document.createElement("div");
    line.className = "qa-distinct-cosine";
    const comparisons = member.comparison?.comparisons;
    if (!Array.isArray(comparisons)) {
      line.textContent = "Chưa có dữ liệu cosine cho vùng này.";
      return line;
    }
    const best = comparisons.filter(item => item.eligible && Number.isFinite(item.similarity))
      .reduce((highest, item) => !highest || item.similarity > highest.similarity ? item : highest, null);
    if (!best) {
      line.textContent = comparisons.length ? "Chưa có vùng trước đó đủ điều kiện so sánh." : "Vùng đầu tiên, chưa có vùng để so sánh.";
      return line;
    }
    const score = document.createElement("span");
    score.textContent = `Cosine cao nhất: ${best.similarity.toFixed(4)} · ${best.above_threshold ? "đạt ngưỡng" : "dưới ngưỡng"} với `;
    const reference = regionButton(best, `${best.image_id} · ${best.label}`, estimate, evidenceByImage);
    line.append(score, reference);
    return line;
  }

  function regroupDistinctEstimate(estimate, threshold) {
    const members = (estimate.groups || []).flatMap(group => group.members || [])
      .sort((a, b) => a.sequence - b.sequence);
    if (!members.length || members.some(member => !member.comparison || !Number.isFinite(member.sequence))) return null;
    const groups = [];
    for (const original of members) {
      const previous = new Map((original.comparison.comparisons || []).map(item => [item.sequence, item]));
      const comparisons = [];
      const options = [];
      groups.forEach((group, index) => {
        const eligible = !group.some(item => item.image_id === original.image_id);
        const scores = [];
        for (const item of group) {
          const prior = previous.get(item.sequence);
          if (!prior || !Number.isFinite(prior.similarity)) continue;
          const comparison = {...prior, group_id: `G${index + 1}`, eligible, above_threshold: prior.similarity >= threshold};
          comparisons.push(comparison);
          scores.push(comparison.similarity);
        }
        if (eligible && scores.some(score => score >= threshold)) options.push({index, score: Math.max(...scores)});
      });
      let index;
      if (options.length) index = options.reduce((best, item) => item.score > best.score ? item : best).index;
      else { index = groups.length; groups.push([]); }
      const decision = options.length ? "matched" : original.sequence === 1 ? "first" : "new_group";
      groups[index].push({...original, comparison: {decision, group_id: `G${index + 1}`, comparisons}});
    }
    return {...estimate, threshold, estimated_count: groups.length,
      groups: groups.map((members, index) => ({group_id: `G${index + 1}`, members})), preview: true};
  }

  // Object browser: every compared region, ordered so that the most similar crops sit next to each other
  // (also across groups), with the current crop and the next one shown large side by side. The reviewer
  // judges whether two neighbours are really the same object, which is where the model's grouping errs.
  const objectViewer = { root: null, order: [], sims: new Map(), groupOf: new Map(), index: 0,
                         estimate: null, evidenceByImage: new Map() };

  function objectViewerItems(estimate) {
    return (estimate.groups || []).flatMap((group, groupIndex) => (group.members || [])
      .filter(member => Array.isArray(member.box))
      .map((member, memberIndex) => ({member, groupId: group.group_id, groupIndex,
        memberIndex, groupSize: group.members.length})));
  }

  function regionSvg(imageId, box, {pad = 0, outline = false, label = ""} = {}) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", label || `Vùng trong ảnh ${imageId}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    const image = document.createElementNS("http://www.w3.org/2000/svg", "image");
    const source = "/images/" + encodeURIComponent(imageId);
    image.setAttribute("href", source);
    svg.appendChild(image);
    const rect = outline ? document.createElementNS("http://www.w3.org/2000/svg", "rect") : null;
    if (rect) {
      rect.setAttribute("class", "object-viewer-box");
      rect.setAttribute("vector-effect", "non-scaling-stroke");
      svg.appendChild(rect);
    }
    const dimensions = new Image();
    dimensions.onload = () => {
      const w = dimensions.naturalWidth, h = dimensions.naturalHeight;
      const [x1, y1, x2, y2] = box;
      const dx = (x2 - x1) * pad, dy = (y2 - y1) * pad;
      const vx1 = Math.max(0, x1 - dx), vy1 = Math.max(0, y1 - dy);
      const vx2 = Math.min(1, x2 + dx), vy2 = Math.min(1, y2 + dy);
      image.setAttribute("width", String(w));
      image.setAttribute("height", String(h));
      svg.setAttribute("viewBox", `${vx1 * w} ${vy1 * h} ${(vx2 - vx1) * w} ${(vy2 - vy1) * h}`);
      if (rect) {
        rect.setAttribute("x", String(x1 * w));
        rect.setAttribute("y", String(y1 * h));
        rect.setAttribute("width", String((x2 - x1) * w));
        rect.setAttribute("height", String((y2 - y1) * h));
      }
    };
    dimensions.src = source;
    return svg;
  }

  // Pairwise cosine from the stored comparisons (each region was compared with every earlier one).
  function objectSimilarities(items) {
    const sims = new Map();
    const bySequence = new Map(items.map(item => [item.member.sequence, item]));
    for (const {member} of items) {
      for (const other of member.comparison?.comparisons || []) {
        if (!bySequence.has(other.sequence) || !Number.isFinite(other.similarity)) continue;
        const sameImage = other.image_id === member.image_id;
        sims.set(`${member.sequence}~${other.sequence}`, {value: other.similarity, sameImage});
        sims.set(`${other.sequence}~${member.sequence}`, {value: other.similarity, sameImage});
      }
    }
    return sims;
  }

  function ensureObjectViewer() {
    if (objectViewer.root) return objectViewer.root;
    const root = document.createElement("div");
    root.className = "object-viewer";
    root.hidden = true;
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.setAttribute("aria-label", "Xem các đối tượng");
    root.innerHTML = `
      <div class="object-viewer-panel">
        <header class="object-viewer-head">
          <div><strong class="object-viewer-title"></strong><span class="object-viewer-count"></span></div>
          <button type="button" class="object-viewer-close" aria-label="Đóng">×</button>
        </header>
        <div class="object-single">
          <button type="button" class="object-viewer-nav" data-step="-1" aria-label="Đối tượng trước">‹</button>
          <figure class="object-single-card"></figure>
          <button type="button" class="object-viewer-nav" data-step="1" aria-label="Đối tượng sau">›</button>
        </div>
        <div class="object-matches-head"></div>
        <div class="object-strip" aria-label="Các vùng có cosine từ ngưỡng trở lên"></div>
        <p class="object-viewer-hint">← → đối tượng khác · bấm một vùng bên dưới để xem lớn · Esc đóng</p>
      </div>`;
    root.addEventListener("click", (event) => {
      if (event.target === root || event.target.closest(".object-viewer-close")) {
        closeObjectViewer();
        return;
      }
      const step = event.target.closest("[data-step]")?.dataset.step;
      if (step) showObjectViewerItem(objectViewer.index + Number(step));
      if (event.target.closest("[data-open]")) {
        const {member} = objectViewer.order[objectViewer.index];
        const evidence = objectViewer.evidenceByImage.get(member.image_id)
          || {image_id: member.image_id, regions: [{box: member.box, label: member.label}]};
        closeObjectViewer();
        createEvidenceActionBtn(evidence, "").click();
      }
    });
    // Capture phase: arrows and Esc belong to the browser while it is open, not to the lightbox behind it.
    document.addEventListener("keydown", (event) => {
      if (root.hidden) return;
      const keys = {ArrowLeft: -1, ArrowRight: 1};
      if (event.key === "Escape") closeObjectViewer();
      else if (event.key in keys) showObjectViewerItem(objectViewer.index + keys[event.key]);
      else return;
      event.preventDefault();
      event.stopImmediatePropagation();
    }, true);
    document.body.appendChild(root);
    objectViewer.root = root;
    return root;
  }

  // Regions that the current threshold links to at least one region of another image.
  function linkedItems(items, sims, threshold) {
    return items.filter(item => items.some(other => {
      const sim = other !== item && sims.get(`${item.member.sequence}~${other.member.sequence}`);
      return sim && !sim.sameImage && sim.value >= threshold;
    }));
  }

  // Greedy chain so that the most similar crops sit next to each other in the strip.
  function similarityChain(items, sims) {
    if (!items.length) return [];
    const left = new Set(items.map((_, i) => i));
    const order = [0];
    left.delete(0);
    while (left.size) {
      const last = items[order[order.length - 1]].member.sequence;
      let best = null, bestValue = -Infinity;
      for (const i of left) {
        const sim = sims.get(`${last}~${items[i].member.sequence}`);
        const value = !sim ? -2 : sim.sameImage ? -1 : sim.value;
        if (value > bestValue) { best = i; bestValue = value; }
      }
      order.push(best);
      left.delete(best);
    }
    return order.map(i => items[i]);
  }

  function openObjectViewer(estimate, member = null, evidenceByImage = new Map()) {
    const root = ensureObjectViewer();
    const items = objectViewerItems(estimate);
    const threshold = Number(estimate.threshold);
    objectViewer.sims = objectSimilarities(items);
    const linked = linkedItems(items, objectViewer.sims, threshold);
    objectViewer.order = similarityChain(linked.length ? linked : items, objectViewer.sims);
    objectViewer.total = items.length;
    objectViewer.estimate = estimate;
    objectViewer.evidenceByImage = evidenceByImage;
    const sameBox = (x, y) => Array.isArray(x) && Array.isArray(y) && x.every((v, i) => Math.abs(v - y[i]) < 1e-6);
    const start = member ? objectViewer.order.findIndex(item => item.member === member
      || (item.member.image_id === member.image_id && sameBox(item.member.box, member.box))) : 0;
    root.querySelector(".object-matches-head").textContent = linked.length
      ? `${linked.length}/${items.length} vùng có ít nhất một vùng khác ảnh với cosine ≥ ${threshold.toFixed(2)} · giống nhau đứng cạnh nhau · kéo để xem`
      : `Không có cặp nào có cosine ≥ ${threshold.toFixed(2)}; đang hiện cả ${items.length} vùng`;
    renderObjectStrip();
    root.hidden = false;
    showObjectViewerItem(Math.max(0, start));
    root.querySelector(".object-viewer-nav[data-step='1']").focus();
  }

  function closeObjectViewer() {
    if (objectViewer.root) objectViewer.root.hidden = true;
  }

  function objectSimilarity(a, b) {
    return a && b ? objectViewer.sims.get(`${a.member.sequence}~${b.member.sequence}`) : null;
  }

  function renderObjectStrip() {
    const strip = objectViewer.root.querySelector(".object-strip");
    const threshold = Number(objectViewer.estimate.threshold);
    strip.replaceChildren(...objectViewer.order.map((item, position) => {
      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = "object-strip-tile";
      tile.title = `${item.groupId} · ${item.member.image_id} · ${item.member.label || ""}`;
      const crop = regionSvg(item.member.image_id, item.member.box, {label: tile.title});
      crop.setAttribute("preserveAspectRatio", "xMidYMid slice");
      const group = document.createElement("span");
      group.className = "object-strip-group";
      group.textContent = item.groupId;
      tile.append(crop, group);
      const sim = objectSimilarity(objectViewer.order[position - 1], item);
      if (sim && !sim.sameImage) {
        const link = document.createElement("span");
        link.className = "object-strip-sim" + (sim.value >= threshold ? " is-above" : "");
        link.textContent = sim.value.toFixed(2);
        link.title = "Cosine với vùng đứng trước";
        tile.appendChild(link);
      }
      tile.addEventListener("click", () => {
        if (strip.dataset.dragged === "1") return;  // the click that ends a drag does not select
        showObjectViewerItem(position);
      });
      return tile;
    }));
    // Drag to scroll (mouse), plus the vertical wheel scrolls the strip sideways.
    if (!strip.dataset.dragReady) {
      strip.dataset.dragReady = "1";
      let startX = 0, startScroll = 0, down = false;
      strip.addEventListener("pointerdown", (event) => {
        if (event.pointerType !== "mouse") return;
        down = true; startX = event.clientX; startScroll = strip.scrollLeft; strip.dataset.dragged = "0";
      });
      window.addEventListener("pointermove", (event) => {
        if (!down) return;
        const dx = event.clientX - startX;
        if (Math.abs(dx) > 4) { strip.dataset.dragged = "1"; strip.classList.add("is-dragging"); }
        strip.scrollLeft = startScroll - dx;
      });
      window.addEventListener("pointerup", () => {
        if (!down) return;
        down = false;
        strip.classList.remove("is-dragging");
        setTimeout(() => { strip.dataset.dragged = "0"; }, 0);
      });
      strip.addEventListener("wheel", (event) => {
        if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
        strip.scrollLeft += event.deltaY;
        event.preventDefault();
      }, {passive: false});
    }
  }

  // The selected region large; in the strip it is framed and every region linked to it (cosine >= t,
  // other image) is marked, so a wrong merge or a missed one shows at a glance.
  function showObjectViewerItem(index) {
    const order = objectViewer.order;
    if (!order.length) return;
    objectViewer.index = (index + order.length) % order.length;
    const root = objectViewer.root;
    const item = order[objectViewer.index];
    const threshold = Number(objectViewer.estimate.threshold);
    root.querySelector(".object-viewer-title").textContent = `${item.groupId} · ${item.member.image_id}`;
    root.querySelector(".object-viewer-count").textContent =
      ` · ${objectViewer.index + 1}/${order.length} · ${objectViewer.estimate.estimated_count} nhóm · t = ${threshold.toFixed(2)}`;
    const card = root.querySelector(".object-single-card");
    const crop = regionSvg(item.member.image_id, item.member.box, {pad: .12, outline: true,
      label: `Vùng ${item.member.label} trong ảnh ${item.member.image_id}`});
    const source = document.createElement("button");
    source.type = "button";
    source.className = "object-single-source";
    source.dataset.open = "1";
    source.title = "Mở ảnh gốc";
    source.appendChild(regionSvg(item.member.image_id, item.member.box, {pad: 99, outline: true,
      label: `Ảnh nguồn ${item.member.image_id}`}));
    card.replaceChildren(crop, source);
    root.querySelectorAll(".object-strip-tile").forEach((tile, i) => {
      const other = order[i];
      const sim = objectSimilarity(item, other);
      tile.setAttribute("aria-current", String(i === objectViewer.index));
      tile.classList.toggle("is-match", Boolean(sim && !sim.sameImage && sim.value >= threshold));
      if (i === objectViewer.index) tile.scrollIntoView({block: "nearest", inline: "center"});
    });
  }

  // A region of a distinct estimate opens the object browser; any other region keeps the image view.
  function regionButton(region, text, estimate = null, evidenceByImage = new Map()) {
    const inEstimate = estimate && objectViewerItems(estimate).some(item => item.member.image_id === region.image_id
      && item.member.box.every((value, i) => Math.abs(value - region.box[i]) < 1e-6));
    if (!inEstimate) return createEvidenceActionBtn({image_id: region.image_id,
      regions: [{box: region.box, label: region.label}]}, text);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "qa-evidence-btn";
    button.textContent = text;
    button.title = "Xem trong danh sách đối tượng";
    button.addEventListener("click", () => openObjectViewer(estimate, region, evidenceByImage));
    return button;
  }

  // The answer only carries one button; everything else is in the object browser.
  // "Xem đối tượng" on every answer: each person or object the answer rests on, cropped from its frame
  // (roster people, counted regions), or the evidence frames themselves when no box was drawn.
  function answerObjects(resp) {
    const evidence = new Map((resp.evidence || []).map(ev => [ev.image_id, ev]));
    if (resp.people_roster?.length) {
      return resp.people_roster.map(person => ({ imageId: person.image_id, box: person.box,
        label: person.label, text: person.appearance,
        ev: evidence.get(person.image_id) || { image_id: person.image_id, regions: person.box ? [{ label: person.label, box: person.box }] : [] } }));
    }
    const items = [];
    for (const ev of evidence.values()) {
      const regions = (ev.regions || []).filter(region => Array.isArray(region.box));
      if (regions.length) {
        for (const region of regions) items.push({ imageId: ev.image_id, box: region.box, label: region.label, text: region.note || "", ev });
      } else {
        items.push({ imageId: ev.image_id, box: null, label: ev.image_id.replace(/\.jpg$/i, ""), text: ev.caption || ev.observation || "", ev });
      }
    }
    return items;
  }

  function createObjectsPanel(resp) {
    // One grid: each object as a tight crop. Duplicate checks live on the tiles: a merged person sits next
    // to the person it repeats (marked "= P5"), a suspected pair carries a "nghi trùng" chip that lights
    // up both tiles.
    const all = answerObjects(resp);
    const pairs = resp.people_duplicates || [];
    const mergedInto = new Map(pairs.filter(p => p.merged).map(p => [p.b, p.a]));
    const suspects = new Map();
    for (const pair of pairs.filter(p => !p.merged)) {
      for (const [self, other] of [[pair.a, pair.b], [pair.b, pair.a]]) {
        if (!suspects.has(self)) suspects.set(self, []);
        suspects.get(self).push({ other, pair });
      }
    }
    const boxed = all.filter(item => item.box);
    const unboxed = all.length - boxed.length;
    // keep each merged person right after the one it repeats
    const order = boxed.filter(item => !mergedInto.has(item.label));
    for (const item of boxed.filter(item => mergedInto.has(item.label))) {
      const at = order.findIndex(other => other.label === mergedInto.get(item.label));
      order.splice(at < 0 ? order.length : at + 1, 0, item);
    }
    const people = Boolean(resp.people_roster?.length);
    const noun = people ? "người" : "đối tượng";
    const panel = document.createElement("details");
    panel.className = "qa-objects-panel";
    const title = document.createElement("summary");
    title.className = "qa-objects-btn";
    const facts = [`${all.length - mergedInto.size} ${noun}`];
    if (mergedInto.size) facts.push(`gộp ${mergedInto.size} trùng`);
    if (suspects.size) facts.push(`${pairs.length - mergedInto.size} cặp nghi trùng`);
    if (unboxed) facts.push(`${unboxed} chưa có khung`);
    title.textContent = boxed.length ? `Xem đối tượng · ${facts.join(" · ")}` : "Xem đối tượng · chưa có đối tượng nào được khoanh vùng";
    panel.appendChild(title);
    const grid = document.createElement("div");
    grid.className = "qa-objects-grid";
    const tiles = new Map();
    const highlight = (labels) => {
      for (const [label, tile] of tiles) tile.classList.toggle("is-paired", labels.includes(label));
    };
    for (const item of order) {
      const tile = document.createElement("div");
      tile.className = "qa-object-tile" + (mergedInto.has(item.label) ? " is-duplicate" : "");
      tile.title = `${item.label} · ${item.imageId}${item.text ? `\n${item.text}` : ""}`;
      const crop = document.createElement("button");
      crop.type = "button";
      crop.className = "qa-object-crop";
      crop.setAttribute("aria-label", `Mở ${item.label} trong ${item.imageId}`);
      crop.appendChild(regionSvg(item.imageId, item.box, { pad: 0, label: `${item.label} trong ${item.imageId}` }));
      crop.addEventListener("click", (event) => { event.stopPropagation(); openLightbox(item.imageId, null, item.ev); });
      const caption = document.createElement("span");
      caption.className = "qa-object-caption";
      const name = document.createElement("b");
      name.textContent = mergedInto.has(item.label) ? `${item.label} = ${mergedInto.get(item.label)}` : item.label;
      caption.append(name, item.text ? ` ${item.text}` : "");
      tile.append(crop, caption);
      for (const { other, pair } of suspects.get(item.label) || []) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "qa-dup-chip";
        chip.textContent = `nghi trùng ${other}`;
        chip.title = `re-ID ${pair.reid ?? "—"} · thuộc tính ${pair.attributes}`;
        chip.addEventListener("click", (event) => { event.stopPropagation(); highlight([item.label, other]); });
        tile.appendChild(chip);
      }
      tiles.set(item.label, tile);
      grid.appendChild(tile);
    }
    panel.appendChild(grid);
    return panel;
  }
  function renderRetrievalPanel(turn, data = {}) {
    if (turn !== activeRetrievalTurn) return;
    const scope = data.examined_scope || {};
    const examined = new Set(scope.image_ids || data.examined_ids || []);
    const evidence = new Map((data.evidence || []).map(ev => [ev.image_id, ev]));
    const ids = [...new Set([...(scope.retrieved_image_ids || data.image_ids || []), ...examined, ...evidence.keys()])];
    const pinnedPreview = !turn && activeQaImageId;
    if (pinnedPreview) ids.push(pinnedPreview);
    const confirmed = data.status === "answered";
    $("qa-retrieval-panel").classList.toggle("has-turn", Boolean(turn || pinnedPreview));
    const coverage = scope.coverage;
    const summary = $("qa-retrieval-summary");
    summary.hidden = !turn;
    summary.textContent = coverage?.source === "reviewed_catalog"
      ? `${coverage.reviewed_images} ảnh đã kiểm duyệt · ${evidence.size} ảnh bằng chứng`
      : coverage?.indexed_images
        ? `${coverage.indexed_images}/${coverage.total_images} ảnh có dữ liệu phát hiện · ${examined.size} ảnh đã đối chiếu`
      : ids.length
      ? `${ids.length} ảnh tìm được · ${examined.size} ảnh đã đọc${confirmed ? ` · ${evidence.size} ảnh bằng chứng` : ""}`
      : turn ? (data.status === "error" ? "Truy xuất chưa hoàn tất." : data.status ? "Chưa có ảnh phù hợp." : "Đang tìm ảnh…") : "";
    $("qa-retrieval-empty").hidden = Boolean(turn || pinnedPreview);
    $("qa-legend").hidden = true;  // the groups below name each state
    const grid = $("qa-retrieval-grid");
    // Per-image verdict from the reader (kept on the turn after the answer arrives).
    const verdict = new Map((data.observations || []).map(o => [o.image_id, o]));
    if (turn && ids.length && !coverage?.source && !coverage?.indexed_images) {
      const matches = [...verdict.values()].filter(o => o.context_match === "yes").length;
      summary.textContent = `${ids.length} ứng viên · đã đọc ${Math.max(examined.size, verdict.size)} · ${matches} khớp cảnh`
        + (confirmed && evidence.size ? ` · ${evidence.size} bằng chứng` : "");
    }
    const key = JSON.stringify([ids, [...examined], [...evidence.keys()], confirmed, pinnedPreview,
                                [...verdict].map(([id, o]) => [id, o.context_match])]);
    if (grid.dataset.contentKey === key) return;
    grid.dataset.contentKey = key;
    grid.replaceChildren();
    const rank = new Map(ids.map((id, index) => [id, index]));
    const groups = pinnedPreview ? [["pinned", "Ảnh đang hỏi", ids, true]] : (() => {
      // Every image the answer cites is its evidence (some answers, like the people roster, are not re-verified).
      const used = confirmed ? ids.filter(id => evidence.has(id)) : [];
      const matching = ids.filter(id => !used.includes(id) && verdict.get(id)?.context_match === "yes");
      const rejected = ids.filter(id => !used.includes(id) && !matching.includes(id) &&
                                        (examined.has(id) || verdict.has(id) || evidence.has(id)));
      const unread = ids.filter(id => !used.includes(id) && !matching.includes(id) && !rejected.includes(id));
      return [["evidence", "Bằng chứng của câu trả lời", used, true],
              ["match", confirmed ? "Cũng khớp cảnh" : "Khớp cảnh (đang xét)", matching, true],
              ["read", "Đã đọc, không khớp", rejected, false],
              ["unread", "Ứng viên chưa đọc", unread, false]];
    })();
    for (const [state, title, members, open] of groups) {
      if (!members.length) continue;
      const section = document.createElement("details");
      section.className = `qa-retrieval-group group-${state}`;
      section.open = open;
      const heading = document.createElement("summary");
      heading.textContent = `${title} · ${members.length}`;
      const cards = document.createElement("div");
      cards.className = "qa-retrieval-cards";
      for (const id of members) {
        const ev = evidence.get(id);
        const seen = verdict.get(id);
        const figure = document.createElement("figure");
        figure.className = "qa-retrieval-card" + (state === "evidence" ? " is-evidence" : "");
        figure.dataset.state = state;
        const button = createEvidenceActionBtn(ev || { image_id: id, regions: seen?.regions || [] }, "");
        button.setAttribute("aria-label", `Xem ảnh ${id}`);
        const img = document.createElement("img");
        img.src = "/thumbs/" + encodeURIComponent(id);
        img.alt = ev?.caption || id;
        img.loading = "lazy";
        img.addEventListener("error", () => { img.src = "/images/" + encodeURIComponent(id); }, { once: true });
        button.appendChild(img);
        const caption = document.createElement("figcaption");
        const name = document.createElement("span");
        name.textContent = pinnedPreview ? id : `#${rank.get(id) + 1} · ${id.replace(/\.jpg$/i, "")}`;
        caption.append(name);
        // One short line under the images that matter; the rest stay as bare thumbnails.
        const note = state === "evidence" ? ev?.caption : state === "match" ? seen?.observation : "";
        if (note) {
          const line = document.createElement("small");
          line.textContent = note;
          line.title = note;
          caption.append(line);
        }
        figure.append(button, caption);
        if (note) figure.title = note;
        cards.appendChild(figure);
      }
      section.append(heading, cards);
      grid.appendChild(section);
    }
  }

  function selectRetrievalTurn(turn) {
    activeRetrievalTurn = turn;
    document.querySelectorAll(".qa-turn").forEach(el => el.classList.toggle("is-selected", el === turn?.turnEl));
    renderRetrievalPanel(turn, turn?.retrievalData || {});
  }

  function appendChatTurn(question, waitingText) {
    if (qaChatEmpty) qaChatEmpty.hidden = true;
    const turnEl = document.createElement("div");
    turnEl.className = "qa-turn";

    const userLine = document.createElement("p");
    userLine.className = "qa-turn-line user-line";
    const userSpeaker = document.createElement("span");
    userSpeaker.className = "qa-speaker user";
    userSpeaker.textContent = "Bạn";
    const userText = document.createElement("span");
    userText.textContent = question;
    userLine.appendChild(userSpeaker);
    userLine.appendChild(userText);

    const botLine = document.createElement("p");
    botLine.className = "qa-turn-line bot-line waiting";
    const botSpeaker = document.createElement("span");
    botSpeaker.className = "qa-speaker bot";
    botSpeaker.textContent = "Castle QA";
    const botText = document.createElement("span");
    botText.textContent = waitingText;
    botLine.appendChild(botSpeaker);
    botLine.appendChild(botText);

    const altContainer = document.createElement("div");
    altContainer.className = "qa-alternatives";
    altContainer.hidden = true;

    const footer = document.createElement("div");
    footer.className = "qa-turn-footer";
    footer.hidden = true;

    turnEl.appendChild(userLine);
    const reply = document.createElement("div");
    reply.className = "qa-assistant-reply";

    // Live status: steps, elapsed time, images read and a stop button.
    const live = document.createElement("div");
    live.className = "qa-live";
    const steps = document.createElement("ol");
    steps.className = "qa-steps";
    steps.setAttribute("aria-label", "Tiến độ xử lý");
    const liveMeta = document.createElement("div");
    liveMeta.className = "qa-live-meta";
    const clock = document.createElement("span");
    clock.className = "qa-live-clock";
    clock.textContent = "0:00";
    const liveCount = document.createElement("span");
    liveCount.className = "qa-live-count";
    const stop = document.createElement("button");
    stop.type = "button";
    stop.className = "qa-stop-btn";
    stop.textContent = "Dừng";
    liveMeta.append(clock, liveCount, stop);
    live.append(steps, liveMeta);

    const badgeRow = document.createElement("div");
    badgeRow.className = "qa-answer-meta";
    badgeRow.hidden = true;
    const caveat = document.createElement("div");
    caveat.className = "qa-caveat";
    caveat.hidden = true;

    const progressDetails = document.createElement("details");
    progressDetails.className = "qa-processing";
    progressDetails.hidden = true;
    const progressTitle = document.createElement("summary");
    progressTitle.textContent = "Chi tiết xử lý";
    const progressList = document.createElement("div");
    progressList.className = "qa-processing-body";
    progressDetails.append(progressTitle, progressList);
    reply.append(botLine, badgeRow, live, caveat, progressDetails, altContainer, footer);
    turnEl.appendChild(reply);

    if (qaChatHistory) {
      qaChatHistory.appendChild(turnEl);
      scrollChatToBottom();
    }

    const turn = { turnEl, botLine, botText, altContainer, footer, progressDetails, progressList, question,
      badgeRow, caveat, live, steps, clock, liveCount, stop, started: performance.now(), stages: [],
      currentStage: null, retrievalData: {} };
    turn.clockTimer = setInterval(() => { clock.textContent = formatElapsed(performance.now() - turn.started); }, 1000);
    renderQaSteps(turn);
    stop.addEventListener("click", (event) => {
      event.stopPropagation();
      if (turn.onStop) turn.onStop();
    });
    turnEl.addEventListener("click", () => selectRetrievalTurn(turn));
    selectRetrievalTurn(turn);
    return turn;
  }

  function formatElapsed(ms) {
    const total = Math.max(0, Math.round(ms / 1000));
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
  }

  const QA_STEPS = [
    ["planning", "Phân tích"], ["retrieving", "Tìm ảnh"], ["observing", "Đọc ảnh"],
    ["synthesizing", "Tổng hợp"], ["verifying", "Kiểm tra"], ["embedding", "Ghép nhóm"],
  ];
  // A draft answer between batches is part of reading the images.
  const QA_STAGE_ALIAS = { scanning: "retrieving", ranking: "retrieving", kis: "retrieving", drafting: "observing", choosing: "observing", rostering: "observing" };

  function renderQaSteps(turn) {
    const seen = new Set(turn.stages.map(stage => QA_STAGE_ALIAS[stage] || stage));
    const current = QA_STAGE_ALIAS[turn.currentStage] || turn.currentStage;
    const finished = turn.currentStage === "done";
    const currentIndex = QA_STEPS.findIndex(([key]) => key === current);
    turn.steps.replaceChildren();
    turn.steps.hidden = !turn.stages.length && !turn.pinnedMode;
    for (const [index, [key, label]] of QA_STEPS.entries()) {
      // Grouping only runs for distinct counts; hide it until it appears.
      if (key === "embedding" && !seen.has(key)) continue;
      const item = document.createElement("li");
      const state = finished || (seen.has(key) && key !== current) || (currentIndex > index)
        ? "done" : key === current ? "active" : "todo";
      item.dataset.state = state;
      if (state === "active") item.setAttribute("aria-current", "step");
      item.textContent = label;
      turn.steps.appendChild(item);
    }
  }

  function finishQaLive(turn, outcome) {
    clearInterval(turn.clockTimer);
    const elapsed = formatElapsed(performance.now() - turn.started);
    turn.live.classList.add("is-finished");
    turn.live.classList.toggle("is-error", outcome === "error");
    turn.stop.hidden = true;
    if (outcome !== "error") turn.currentStage = "done";
    renderQaSteps(turn);
    turn.clock.textContent = outcome === "error" ? `Dừng sau ${elapsed}` : `Hoàn tất trong ${elapsed}`;
  }

  const QA_SCOPE_NAMES = { scene: "một cảnh", per_image: "từng ảnh", comparison: "so sánh ảnh", corpus: "toàn bộ tập ảnh" };
  const QA_OPERATIONS = { count_distinct_people: "đếm người khác nhau", count_distinct_objects: "đếm đối tượng khác nhau",
    count_visible: "đếm đối tượng nhìn thấy", count_max: "tìm số lượng lớn nhất", describe: "mô tả", find: "tìm đối tượng",
    compare: "so sánh", count_events: "đếm sự kiện", temporal_relation: "xét thứ tự thời gian" };

  function renderQaProcessing(turn, progress) {
    turn.stages = [...new Set(progress.stages || [])];
    turn.currentStage = progress.stage && progress.stage !== "idle" ? progress.stage : turn.currentStage;
    renderQaSteps(turn);
    const read = (progress.examined_ids || []).length;
    const found = (progress.image_ids || []).length;
    if (found || read) turn.liveCount.textContent = `Đã đọc ${read}/${found} ảnh`;
    if (!progress.plan && !progress.observations?.length) return;
    turn.progressDetails.hidden = false;
    const observationsOpen = Boolean(turn.progressList.querySelector(".qa-observation-details")?.open);
    turn.progressList.replaceChildren();
    if (progress.plan) {
      const plan = progress.plan;
      const block = document.createElement("div");
      block.className = "qa-plan-compact";
      const chips = document.createElement("div");
      chips.className = "qa-plan-chips";
      for (const value of [QA_OPERATIONS[plan.operation] || plan.operation, QA_SCOPE_NAMES[plan.answer_scope] || plan.answer_scope, plan.subject]) {
        if (!value) continue;
        const chip = document.createElement("span");
        chip.textContent = value;
        chips.appendChild(chip);
      }
      block.appendChild(chips);
      for (const [term, value] of [["Hiểu là", plan.question], ["Điều kiện", plan.known_conditions?.join("; ")],
                                   ["Truy vấn", plan.queries?.join("; ")], ["Cần làm rõ", plan.clarification]]) {
        if (!value) continue;
        const line = document.createElement("p");
        const label = document.createElement("b");
        label.textContent = `${term}: `;
        line.append(label, value);
        block.appendChild(line);
      }
      turn.progressList.appendChild(block);
    }
    if (progress.observations?.length) {
      const details = document.createElement("details");
      details.className = "qa-observation-details";
      details.open = observationsOpen;
      const title = document.createElement("summary");
      const fit = progress.observations.filter(o => o.context_match === "yes").length;
      title.textContent = `Mô hình đã xem ${progress.observations.length} ảnh · ${fit} phù hợp`;
      details.appendChild(title);
      const table = document.createElement("div");
      table.className = "qa-observation-table";
      const mark = {yes: ["✓", "phù hợp"], no: ["✗", "không phù hợp"], unclear: ["?", "chưa rõ"]};
      const checkMark = {met: "✓", violated: "✗", unclear: "?"};
      for (const observation of progress.observations) {
        const row = document.createElement("div");
        row.className = `qa-observation-line is-${observation.context_match || "unclear"}`;
        const open = createEvidenceActionBtn({image_id: observation.image_id, regions: observation.regions || []},
                                             observation.image_id.replace(/\.jpg$/i, ""));
        open.title = "Xem ảnh và các vùng";
        const [symbol, word] = mark[observation.context_match] || mark.unclear;
        const status = document.createElement("span");
        status.className = "qa-observation-status";
        status.textContent = symbol;
        status.title = word;
        const count = document.createElement("span");
        count.className = "qa-observation-count";
        count.textContent = observation.regions?.length
          ? `${observation.regions.length}${observation.count_visibility === "complete" ? "" : "+"} vùng` : "";
        const text = document.createElement("span");
        text.className = "qa-observation-text";
        const checks = (observation.checks || []).map(check => `ĐK${check.condition_id} ${checkMark[check.status] || "?"}`).join(" ");
        text.textContent = [observation.observation, checks, observation.uncertainty && `chưa chắc: ${observation.uncertainty}`]
          .filter(Boolean).join(" · ") || "—";
        text.title = text.textContent;
        row.append(status, open, count, text);
        // Rejected frames stay out of the way until asked for.
        if (observation.context_match === "no") row.hidden = !turn.showRejected;
        table.appendChild(row);
      }
      details.appendChild(table);
      const rejectedCount = progress.observations.filter(o => o.context_match === "no").length;
      if (rejectedCount) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "qa-copy-btn qa-rejected-toggle";
        toggle.textContent = turn.showRejected ? `Ẩn ${rejectedCount} ảnh không khớp` : `Hiện ${rejectedCount} ảnh không khớp`;
        toggle.addEventListener("click", (event) => {
          event.stopPropagation();
          turn.showRejected = !turn.showRejected;
          table.querySelectorAll(".qa-observation-line.is-no").forEach(row => { row.hidden = !turn.showRejected; });
          toggle.textContent = turn.showRejected ? `Ẩn ${rejectedCount} ảnh không khớp` : `Hiện ${rejectedCount} ảnh không khớp`;
        });
        details.appendChild(toggle);
      }
      turn.progressList.appendChild(details);
    }
  }

  function distinctNoun(estimate, lang) {
    const category = String(estimate?.category || "").toLowerCase();
    if (["person", "people", "người"].includes(category)) return lang === "en" ? "people" : "người";
    return category || (lang === "en" ? "objects" : "đối tượng");
  }

  // Largest number of regions in one image: a lower bound that does not depend on t.
  function distinctLowerBound(estimate) {
    const perImage = new Map();
    for (const group of estimate.groups || []) {
      for (const member of group.members || []) perImage.set(member.image_id, (perImage.get(member.image_id) || 0) + 1);
    }
    let best = null;
    for (const [imageId, count] of perImage) if (!best || count > best.count) best = { imageId, count };
    return best;
  }

  // Rebuild the answer from the grouping at the current threshold.
  function composeDistinctAnswer(estimate, resp, lang) {
    const noun = distinctNoun(estimate, lang);
    const count = estimate.estimated_count;
    const read = estimate.examined_images;
    const planned = estimate.requested_images || read;
    const bound = distinctLowerBound(estimate);
    const en = lang === "en";
    const main = en
      ? `Estimated ${count} distinct ${noun} across ${read} examined image${read === 1 ? "" : "s"}` +
        (bound && bound.count > 1 ? `; at least ${bound.count} appear together in ${bound.imageId}.` : ".")
      : `Ước lượng ${count} ${noun} khác nhau trong ${read} ảnh đã đọc` +
        (bound && bound.count > 1 ? `; chắc chắn có ít nhất ${bound.count} ${noun} cùng xuất hiện trong ảnh ${bound.imageId}.` : ".");
    const caveats = [];
    caveats.push(en
      ? `Grouped by cosine similarity ≥ t = ${estimate.threshold.toFixed(2)}; similar-looking ${noun} may be merged and the same one may be split.`
      : `Ghép nhóm theo cosine ≥ t = ${estimate.threshold.toFixed(2)}; ${noun} trông giống nhau có thể bị gộp, cùng một ${noun} có thể bị tách.`);
    if (estimate.stopped_early && read < planned) {
      caveats.push(en ? `Stopped after ${read}/${planned} images: the last images added no new ${noun}.`
        : `Dừng sớm sau ${read}/${planned} ảnh vì các ảnh cuối không thêm ${noun} mới.`);
    } else if (read < planned || estimate.incomplete) {
      caveats.push(en ? `Only ${read}/${planned} planned images were read before the time limit.`
        : `Mới đọc ${read}/${planned} ảnh dự kiến trước khi hết thời gian.`);
    }
    if (estimate.unassigned_regions) {
      caveats.push(en ? `${estimate.unassigned_regions} regions could not be compared and were excluded.`
        : `${estimate.unassigned_regions} vùng chưa so sánh được nên chưa tính.`);
    }
    caveats.push(en ? "Not a total for the whole dataset." : "Chưa phải tổng số trên toàn bộ dữ liệu.");
    if (resp.status !== "answered" && resp.message) {
      const reason = resp.message.split(/\s*(?:Ước lượng thử nghiệm|Experimental estimate)/)[0].trim();
      if (reason) caveats.unshift(reason);
    }
    const short = bound && bound.count > 1
      ? (en ? `About ${count} (at least ${bound.count})` : `Khoảng ${count} ${noun} (ít nhất ${bound.count})`)
      : (en ? `About ${count}` : `Khoảng ${count} ${noun}`);
    return { main, caveats, short };
  }

  function answerBadge(resp) {
    if (resp.status === "answered" && resp.distinct_estimate) return ["estimate", "Ước lượng · chưa xác minh"];
    if (resp.status === "answered" && resp.partial) return ["partial", "Trả lời một phần"];
    if (resp.status === "answered") {
      const verified = (resp.evidence || []).some(ev => ev.verification_status === "verified");
      return ["answered", verified ? "Đã kiểm tra bằng chứng" : "Đã trả lời"];
    }
    if (resp.status === "ambiguous") return ["ambiguous", "Cần làm rõ"];
    return ["insufficient_evidence", "Chưa đủ bằng chứng"];
  }

  function renderAnswerMeta(turn, resp, estimate = null) {
    const [state, label] = answerBadge(resp);
    const scope = resp.examined_scope || {};
    const read = (scope.image_ids || []).length || resp.examined_count || 0;
    const found = (scope.retrieved_image_ids || []).length || resp.searched_count || 0;
    const chips = [];
    if (found) chips.push(`${read}/${found} ảnh đã đọc`);
    if (estimate) chips.push(`${estimate.estimated_count} nhóm · t = ${estimate.threshold.toFixed(2)}`);
    if (scope.stopped_early) chips.push("Dừng sớm · đủ bằng chứng");
    if (resp.api_calls) chips.push(`${resp.api_calls} lượt gọi mô hình`);
    if (resp.cache_hit) chips.push("từ bộ nhớ đệm");
    const badge = document.createElement("span");
    badge.className = `qa-badge ${state}`;
    badge.textContent = label;
    turn.badgeRow.replaceChildren(badge, ...chips.map(text => {
      const chip = document.createElement("span");
      chip.className = "qa-chip";
      chip.textContent = text;
      return chip;
    }));
    turn.badgeRow.hidden = false;
  }

  function renderCaveats(turn, caveats) {
    turn.caveat.replaceChildren();
    const items = caveats.filter(Boolean);
    turn.caveat.hidden = !items.length;
    if (!items.length) return;
    const list = document.createElement("ul");
    for (const text of items) {
      const li = document.createElement("li");
      li.textContent = text;
      list.appendChild(li);
    }
    turn.caveat.appendChild(list);
  }

  // Threshold slider: regroups the stored comparisons and regenerates the answer.
  function finalizeChatTurn(turn, resp, lang, fallbackImageId = null) {
    turn.botLine.classList.remove("waiting", "error");
    finishQaLive(turn, "done");
    const shortDefaultInsufficient =
      lang === "en" ? "Image is not clear enough to determine." : "Ảnh chưa đủ rõ để xác định.";
    let displayText =
      resp.status === "answered"
        ? resp.answer || shortDefaultInsufficient
        : resp.message || shortDefaultInsufficient;
    // Text sent to DRES: the bare answer ("4 người", "màu nâu") when the backend has one, else the full answer.
    turn.submitText = resp.status === "answered" ? (resp.short_answer || resp.answer) : null;
    const estimate = resp.distinct_estimate?.groups?.length ? resp.distinct_estimate : null;
    if (estimate) {
      const composed = composeDistinctAnswer(estimate, resp, lang);
      displayText = composed.main;
      // An estimate the backend did not accept (insufficient_evidence) is shown, never offered for DRES.
      turn.submitText = resp.status === "answered" ? composed.short : null;
      renderCaveats(turn, composed.caveats);
    } else if (resp.status === "answered" && resp.partial) {
      renderCaveats(turn, ["Chỉ phần đã xác nhận được trả lời; các ảnh còn lại chưa đủ bằng chứng."]);
    }
    turn.botText.textContent = displayText;
    turn.displayText = displayText;
    renderAnswerMeta(turn, resp, estimate);
    turn.footer.hidden = false;
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "qa-copy-btn";
    copy.textContent = "Sao chép";
    copy.setAttribute("aria-label", "Sao chép câu trả lời");
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(turn.displayText);
        copy.textContent = "Đã sao chép";
        setTimeout(() => { copy.textContent = "Sao chép"; }, 1800);
      } catch (_) { copy.textContent = "Chưa sao chép được"; }
    });
    turn.footer.appendChild(copy);
    if (turn.submitText) {
      const useAnswer = document.createElement("button");
      useAnswer.type = "button";
      useAnswer.className = "qa-copy-btn qa-use-answer";
      useAnswer.textContent = "Dùng làm đáp án";
      useAnswer.hidden = localQaMode;
      useAnswer.addEventListener("click", () => {
        qaSubmitText.value = turn.submitText;
        $("qa-submit-panel").open = true;
        qaSubmitText.focus();
        updateQaSubmitAvailability();
      });
      turn.footer.appendChild(useAnswer);
    }

    turn.retrievalData = { ...resp, observations: resp.observations || turn.retrievalData?.observations || [] };
    if (fallbackImageId && !resp.evidence?.length) {
      turn.retrievalData = { ...resp, image_ids: [fallbackImageId] };
    }
    renderRetrievalPanel(turn, turn.retrievalData);
    // Show cited images with the answer; the right panel retains all retrieval candidates.
    const answerEvidence = [...new Map((resp.evidence || []).map(ev => [ev.image_id, ev])).values()];
    if (answerEvidence.length) {
      const sources = document.createElement("section");
      sources.className = "qa-answer-sources";
      if (resp.distinct_estimate) sources.classList.add("qa-sources-compact");
      sources.setAttribute("aria-label", "Ảnh đi kèm câu trả lời");
      const heading = document.createElement("p");
      heading.className = "qa-sources-title";
      heading.textContent = resp.status === "answered" && resp.partial
        ? "Ảnh hỗ trợ phần đã xác nhận"
        : resp.status === "answered"
        ? "Ảnh hỗ trợ câu trả lời"
        : "Ảnh liên quan · Chưa đủ bằng chứng để kết luận";
      const gallery = document.createElement("div");
      gallery.className = "qa-inline-evidence";
      const scope = resp.examined_scope || {};
      const orderedIds = [...new Set([...(scope.retrieved_image_ids || []), ...(scope.image_ids || []), ...answerEvidence.map(ev => ev.image_id)])];
      answerEvidence.forEach((ev, i) => {
        const number = orderedIds.indexOf(ev.image_id) + 1;
        const figure = document.createElement("figure");
        figure.hidden = i >= 3;
        const button = createEvidenceActionBtn(ev, "");
        button.setAttribute("aria-label", `Xem ảnh bằng chứng ${number}: ${ev.image_id}`);
        const img = document.createElement("img");
        img.src = "/thumbs/" + encodeURIComponent(ev.image_id);
        img.alt = ev.caption || `Ảnh ${number}: ${ev.image_id}`;
        img.loading = "lazy";
        img.addEventListener("error", () => { img.src = "/images/" + encodeURIComponent(ev.image_id); }, { once: true });
        button.appendChild(img);
        const caption = document.createElement("figcaption");
        const id = document.createElement("strong");
        id.textContent = `Ảnh ${number} · ${ev.image_id}`;
        const description = document.createElement("span");
        description.textContent = ev.caption || (resp.claims || []).filter(c => c.evidence_ids?.includes(ev.image_id)).map(c => c.text).join(" ");
        description.title = description.textContent;
        caption.append(id, description);
        figure.append(button, caption);
        gallery.appendChild(figure);
      });
      sources.append(heading, gallery);
      if (answerEvidence.length > 3) {
        const more = document.createElement("button");
        more.type = "button";
        more.className = "qa-copy-btn";
        more.textContent = `Xem thêm ${answerEvidence.length - 3} ảnh hỗ trợ`;
        more.addEventListener("click", () => {
          gallery.querySelectorAll("figure").forEach(figure => { figure.hidden = false; });
          more.hidden = true;
          scrollChatToBottom();
        });
        sources.appendChild(more);
      }
      turn.footer.prepend(sources);
    }
    if (resp.status === "answered" || answerEvidence.length) {
      const objects = createObjectsPanel(resp);
      const sources = turn.footer.querySelector(".qa-answer-sources");
      if (sources) sources.after(objects);
      else turn.footer.prepend(objects);
    }
    if (!resp.distinct_estimate && Array.isArray(resp.identity_match_candidates) && resp.identity_match_candidates.length) {
      const group = document.createElement("details");
      group.className = "qa-answer-sources";
      const heading = document.createElement("summary");
      heading.textContent = "Các cặp vùng người cần đối chiếu";
      const warning = document.createElement("p");
      warning.className = "qa-sources-title";
      warning.textContent = "Chỉ là gợi ý theo độ giống hình ảnh, chưa xác nhận cùng người và không dùng để tính tổng.";
      group.append(heading, warning);
      for (const candidate of resp.identity_match_candidates) {
        const row = document.createElement("p");
        for (const [index, person] of [candidate.first, candidate.second].entries()) {
          if (index) row.append(document.createTextNode(" ↔ "));
          const ev = (resp.evidence || []).find(item => item.image_id === person.image_id)
            || {image_id: person.image_id, regions: [{box: person.box, label: person.label}]};
          row.append(createEvidenceActionBtn(ev, `${person.image_id} · ${person.label}`));
        }
        group.append(row);
      }
      turn.footer.append(group);
    }
    const viewImages = document.createElement("button");
    viewImages.type = "button";
    viewImages.className = "qa-copy-btn";
    viewImages.textContent = "Xem ảnh truy xuất →";
    viewImages.addEventListener("click", () => {
      selectRetrievalTurn(turn);
      if (window.innerWidth <= 820) {
        const behavior = matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
        $("qa-retrieval-panel").scrollIntoView({ behavior, block: "start" });
      }
    });
    turn.footer.appendChild(viewImages);
    if (resp.request_id) turn.turnEl.dataset.requestId = resp.request_id;
    if (resp.conversation_id) qaConversationId = resp.conversation_id;

    if (resp.note) {
      turn.footer.hidden = false;
      const noteEl = document.createElement("div");
      noteEl.className = "qa-turn-note";
      const noteText = document.createElement("span");
      noteText.textContent = resp.note;
      noteEl.appendChild(noteText);

      if (Array.isArray(resp.note_evidence) && resp.note_evidence.length > 0) {
        const altEv = resp.note_evidence[0];
        noteEl.appendChild(createEvidenceActionBtn(altEv, "Xem"));
        const switchBtn = document.createElement("button");
        switchBtn.type = "button";
        switchBtn.className = "qa-link-btn";
        switchBtn.textContent = "Đổi sang ảnh này";
        switchBtn.addEventListener("click", () => {
          setActiveQaImage(altEv.image_id);
          if (retrievalQaQuestion) retrievalQaQuestion.focus();
        });
        noteEl.appendChild(switchBtn);
      }
      turn.footer.appendChild(noteEl);
    }

    if (Array.isArray(resp.alternatives) && resp.alternatives.length > 0) {
      turn.altContainer.hidden = false;
      resp.alternatives.forEach((alt) => {
        const card = document.createElement("div");
        card.className = "qa-alt-item";
        const ansP = document.createElement("p");
        ansP.className = "qa-alt-answer";
        ansP.textContent = alt.answer;
        card.appendChild(ansP);

        if (alt.clarification) {
          const clarP = document.createElement("p");
          clarP.className = "qa-alt-clarification";
          clarP.textContent = alt.clarification;
          card.appendChild(clarP);
        }

        if (Array.isArray(alt.evidence) && alt.evidence.length > 0) {
          const actions = document.createElement("div");
          actions.className = "qa-evidence-actions";
          const primaryEv = alt.evidence[0];
          actions.appendChild(createEvidenceActionBtn(primaryEv, "Xem ảnh bằng chứng"));

          const useImgBtn = document.createElement("button");
          useImgBtn.type = "button";
          useImgBtn.className = "qa-link-btn";
          useImgBtn.textContent = "Hỏi tiếp trên ảnh này";
          useImgBtn.addEventListener("click", () => {
            setActiveQaImage(primaryEv.image_id);
            if (retrievalQaQuestion) retrievalQaQuestion.focus();
          });
          actions.appendChild(useImgBtn);
          card.appendChild(actions);
        }
        turn.altContainer.appendChild(card);
      });
    }

    scrollChatToBottom();
  }

  function failChatTurn(turn, errorMessage) {
    finishQaLive(turn, "error");
    turn.botLine.classList.remove("waiting");
    turn.botLine.classList.add("error");
    turn.botText.textContent = errorMessage;
    turn.retrievalData = { ...turn.retrievalData, status: "error" };
    renderRetrievalPanel(turn, turn.retrievalData);
    scrollChatToBottom();
  }

  function setLightboxQaUiState(state, lang, messageText, metaText = "") {
    lightboxQaState = state;
    lightboxQaResult.hidden = false;
    lightboxQaStatus.className = `qa-badge ${state}`;
    lightboxQaStatus.textContent = statusLabel(state, lang);
    lightboxQaMeta.textContent = metaText;
    lightboxQaAnswer.textContent = messageText || "";
  }

  if (lightboxQaForm) {
    lightboxQaForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!currentImageId) return;
      if (lightboxQaState === "waiting" || lightboxQaState === "validating") return;

      const capturedImageId = currentImageId;
      const lang = lightboxQaLang ? lightboxQaLang.value : "vi";
      const rawQuestion = lightboxQaInput ? lightboxQaInput.value : "";
      const question = rawQuestion.normalize("NFC").trim();
      const maxLen = qaStatusInfo.max_question_length || 1200;

      // Transition: idle -> validating
      lightboxQaEvidence.replaceChildren();
      setLightboxQaUiState("validating", lang, "");

      if (!question) {
        setLightboxQaUiState(
          "error",
          lang,
          lang === "en" ? "Please enter a question." : "Vui lòng nhập câu hỏi."
        );
        return;
      }
      if (question.length > maxLen) {
        setLightboxQaUiState(
          "error",
          lang,
          lang === "en"
            ? `Question exceeds ${maxLen} characters.`
            : `Câu hỏi vượt quá ${maxLen} ký tự.`
        );
        return;
      }

      // Sync active image in chat history so follow-up questions continue on this image
      setActiveQaImage(capturedImageId);
      const chatTurn = appendChatTurn(
        question,
        lang === "en" ? "Checking image…" : "Đang trả lời…"
      );
      chatTurn.liveCount.textContent = `Ảnh ${capturedImageId}`;
      chatTurn.onStop = () => lightboxQaAbort?.abort();

      // Transition: validating -> waiting
      lightboxQaSeq += 1;
      const reqSeq = lightboxQaSeq;
      if (lightboxQaAbort) {
        lightboxQaAbort.abort();
      }
      const controller = new AbortController();
      lightboxQaAbort = controller;
      lightboxQaBtn.disabled = true;
      setLightboxQaUiState(
        "waiting",
        lang,
        lang === "en" ? "Checking image…" : "Đang trả lời…"
      );

      try {
        const resp = await api("/api/qa", {
          method: "POST",
          signal: controller.signal,
          body: JSON.stringify({
            mode: "selected_image",
            question,
            image_id: capturedImageId,
            answer_language: lang,
          }),
        });

        finalizeChatTurn(chatTurn, resp, lang, capturedImageId);

        // Stale response guard: discard lightbox DOM update if image or sequence counter changed
        if (reqSeq !== lightboxQaSeq || currentImageId !== capturedImageId) {
          return;
        }

        const cacheTag = resp.cache_hit ? " · cache" : "";
        const modelTag = resp.versions && resp.versions.model ? ` · ${resp.versions.model}` : "";
        const metaStr = `${resp.took_ms}ms${cacheTag}${modelTag}`;

        const displayText =
          resp.status === "answered"
            ? resp.answer || ""
            : resp.message || "Ảnh chưa đủ rõ để xác định.";

        setLightboxQaUiState(resp.status, lang, displayText, metaStr);
        if (lightboxQaInput) lightboxQaInput.value = "";

        lightboxQaEvidence.replaceChildren();
        if (Array.isArray(resp.evidence)) {
          for (const ev of resp.evidence) {
            if (Array.isArray(ev.box)) {
              const btn = document.createElement("button");
              btn.type = "button";
              btn.className = "qa-evidence-btn";
              btn.textContent = "Xem vùng bằng chứng";
              btn.addEventListener("click", () => {
                highlightEvidenceBox(ev.box, ev.image_id);
              });
              lightboxQaEvidence.appendChild(btn);
            }
          }
        }
      } catch (err) {
        if (err.name === "AbortError") {
          failChatTurn(chatTurn, "Đã hủy câu hỏi.");
          return;
        }
        const errMsg = formatQaErrorMessage(err);
        failChatTurn(chatTurn, errMsg);
        if (reqSeq !== lightboxQaSeq || currentImageId !== capturedImageId) {
          return;
        }
        if (err.status === 401) {
          setLightboxQaUiState(
            "error",
            lang,
            "Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại (câu hỏi của bạn vẫn được giữ)."
          );
          setTimeout(() => {
            lightbox.hidden = true;
            showLoggedOut();
          }, 1200);
          return;
        }
        setLightboxQaUiState("error", lang, errMsg);
      } finally {
        if (reqSeq === lightboxQaSeq) {
          lightboxQaBtn.disabled = false;
          lightboxQaAbort = null;
        }
      }
    });
  }

  // Closing the request alone leaves the server answering; ask it to stop (and, for a new chat, forget).
  function resetQaOnServer(conversationId, clear) {
    api("/api/qa/reset", {
      method: "POST",
      body: JSON.stringify({ conversation_id: conversationId, clear }),
    }).catch(() => { /* Logged out or QA off: nothing is running for this user. */ });
  }

  function abortRetrievalQa() {
    retrievalQaSeq += 1;
    if (retrievalQaAbort) {
      retrievalQaAbort.abort();
      retrievalQaAbort = null;
      resetQaOnServer(qaConversationId, false);
    }
    if (retrievalQaSubmit) {
      retrievalQaSubmit.disabled = false;
    }
  }

  function resizeQaInput() {
    if (!retrievalQaQuestion) return;
    retrievalQaQuestion.style.height = "auto";
    retrievalQaQuestion.style.height = Math.min(retrievalQaQuestion.scrollHeight, 160) + "px";
  }

  document.querySelectorAll("[data-qa-suggestion]").forEach(button => {
    button.addEventListener("click", () => {
      retrievalQaQuestion.value = activeQaImageId ? button.dataset.qaPinned : button.dataset.qaSuggestion;
      resizeQaInput();
      retrievalQaQuestion.focus();
    });
  });

  $("qa-new-chat")?.addEventListener("click", () => {
    abortRetrievalQa();
    resetQaOnServer(qaConversationId, true);
    qaConversationId = crypto.randomUUID();
    setActiveQaImage(null);
    qaChatHistory.replaceChildren(qaChatEmpty);
    qaChatEmpty.hidden = false;
    selectRetrievalTurn(null);
    retrievalQaQuestion.value = "";
    retrievalQaQuery.value = "";
    if (qaHintRow) qaHintRow.open = false;
    resizeQaInput();
    retrievalQaQuestion.focus();
  });

  if (retrievalQaQuestion) {
    retrievalQaQuestion.addEventListener("input", resizeQaInput);
    retrievalQaQuestion.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        if (retrievalQaForm && !retrievalQaSubmit.disabled) {
          retrievalQaForm.requestSubmit();
        }
      }
    });
  }

  if (retrievalQaForm) {
    retrievalQaForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const lang = retrievalQaLang ? retrievalQaLang.value : "vi";
      const question = (retrievalQaQuestion ? retrievalQaQuestion.value : "").normalize("NFC").trim();
      const retrievalQuery = (retrievalQaQuery ? retrievalQaQuery.value : "").normalize("NFC").trim();
      const maxLen = qaStatusInfo.max_question_length || 1200;

      if (!question || question.length > maxLen) {
        const errTurn = appendChatTurn(question || "(trống)", "");
        failChatTurn(
          errTurn,
          !question ? "Vui lòng nhập câu hỏi." : `Câu hỏi vượt quá ${maxLen} ký tự.`
        );
        return;
      }

      // Clear input immediately after accepting the turn
      if (retrievalQaQuestion) {
        retrievalQaQuestion.value = "";
        resizeQaInput();
      }

      const pinnedImageId = activeQaImageId;
      const waitingMsg = pinnedImageId
        ? lang === "en"
          ? "Checking current image…"
          : "Đang trả lời trên ảnh đang xét…"
        : lang === "en"
          ? "Finding relevant image and answering…"
          : "Đang tìm ảnh phù hợp và trả lời…";

      const chatTurn = appendChatTurn(question, waitingMsg);
      if (pinnedImageId) chatTurn.liveCount.textContent = `Ảnh ${pinnedImageId}`;

      retrievalQaSeq += 1;
      const reqSeq = retrievalQaSeq;
      if (retrievalQaAbort) {
        retrievalQaAbort.abort();
      }
      const controller = new AbortController();
      retrievalQaAbort = controller;
      retrievalQaSubmit.disabled = true;
      chatTurn.onStop = () => {
        if (reqSeq !== retrievalQaSeq) return;
        abortRetrievalQa();
        failChatTurn(chatTurn, lang === "en" ? "Stopped." : "Đã dừng. Bạn có thể hỏi lại hoặc thu hẹp câu hỏi.");
        if (retrievalQaQuestion && !retrievalQaQuestion.value) {
          retrievalQaQuestion.value = question;
          resizeQaInput();
        }
      };

      const payload = pinnedImageId
        ? {
            mode: "selected_image",
            question,
            image_id: pinnedImageId,
            answer_language: lang,
          }
        : {
            mode: "auto",
            conversation_id: qaConversationId,
            question,
            answer_language: lang,
            ...(retrievalQuery ? { retrieval_query: retrievalQuery } : {}),
          };

      let progressBusy = false;
      let progressFinished = false;
      const progressTimer = !pinnedImageId ? setInterval(async () => {
        if (progressBusy || controller.signal.aborted) return;
        progressBusy = true;
        try {
          const progress = await api("/api/qa/progress/" + encodeURIComponent(qaConversationId), { signal: controller.signal });
          if (reqSeq !== retrievalQaSeq || progressFinished) return;
          renderQaProcessing(chatTurn, progress);
          const labels = lang === "en"
            ? { planning: "Understanding question…", scanning: "Scanning the image catalog…", retrieving: "Finding images…", ranking: "Ranking candidates by metadata…", kis: "Finding the scene with image search…", rostering: "Listing each distinct person…", choosing: "Comparing the closest frames…", observing: "Examining evidence…", synthesizing: "Combining evidence…", drafting: "Trying a draft answer…", verifying: "Checking answer…" }
            : { planning: "Đang hiểu câu hỏi…", scanning: "Đang rà danh mục ảnh…", retrieving: "Đang tìm ảnh…", ranking: "Đang xếp ứng viên theo metadata…", kis: "Đang tìm cảnh bằng bộ tìm ảnh (KIS)…", rostering: "Đang điểm danh từng người…", choosing: "Đang so sánh các khung gần nhất…", observing: "Đang đối chiếu…", synthesizing: "Đang tổng hợp…", drafting: "Đang thử câu trả lời nháp…", verifying: "Đang kiểm tra câu trả lời…" };
          if (progress.stage === "embedding") labels.embedding = lang === "en" ? "Comparing object crops…" : "Đang so sánh các vùng đối tượng…";
          if (labels[progress.stage]) chatTurn.botText.textContent = labels[progress.stage];
          if (progress.image_ids?.length) {
            chatTurn.retrievalData = progress;
            renderRetrievalPanel(chatTurn, progress);
          }
        } catch (_) { /* The main request reports errors. */ }
        finally { progressBusy = false; }
      }, 1000) : null;
      try {
        const resp = await api("/api/qa", {
          method: "POST",
          signal: controller.signal,
          body: JSON.stringify(payload),
        });
        if (reqSeq !== retrievalQaSeq) return;

        if (!pinnedImageId) {
          try {
            const progress = await api("/api/qa/progress/" + encodeURIComponent(qaConversationId), {signal: controller.signal});
            if (!resp.request_id || progress.request_id === resp.request_id) renderQaProcessing(chatTurn, progress);
          } catch (_) { /* The answer is still usable if progress is unavailable. */ }
        }
        if (reqSeq !== retrievalQaSeq) return;
        finalizeChatTurn(chatTurn, resp, lang, pinnedImageId);
      } catch (err) {
        if (err.name === "AbortError" || reqSeq !== retrievalQaSeq) return;
        if (err.status === 401) {
          // Restore question in input so user does not lose it
          if (retrievalQaQuestion) retrievalQaQuestion.value = question;
          showLoggedOut();
          return;
        }
        failChatTurn(chatTurn, formatQaErrorMessage(err));
      } finally {
        progressFinished = true;
        clearInterval(progressTimer);
        if (reqSeq === retrievalQaSeq) {
          retrievalQaSubmit.disabled = false;
          retrievalQaAbort = null;
          if (retrievalQaQuestion) retrievalQaQuestion.focus();
        }
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    if (!lightbox.hidden && (event.key === "ArrowLeft" || event.key === "ArrowRight")
        && !event.target.closest?.("input, textarea, select, [role=tab]")) {
      event.preventDefault();
      stepNeighbor(event.key === "ArrowLeft" ? -1 : 1);
      return;
    }
    if (event.key === "Escape") {
      confirmDialog.hidden = true;
      closeLightbox();
    }
  });
  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });

  closeLightboxBtn.addEventListener("click", () => {
    closeLightbox();
  });

  submitBtn.addEventListener("click", () => {
    if (!currentImageId) return;
    const note = kisSubmitted.correct ? " Team ĐÃ có bài đúng cho task này, không cần nộp thêm."
      : kisSubmitted.ids.has(currentImageId) ? " Ảnh này đã nộp cho task này rồi (DRES sẽ từ chối)." : "";
    confirmDialog.querySelector("p").textContent = `Nộp ${currentImageId} vào buổi thi ${selectedEvaluationId}? Nộp sai có thể bị trừ điểm.${note}`;
    confirmDialog.hidden = false;
  });

  confirmNo.addEventListener("click", () => {
    confirmDialog.hidden = true;
  });

  confirmYes.addEventListener("click", async () => {
    confirmDialog.hidden = true;
    if (!currentImageId) return;
    const id = currentImageId;
    submitResult.textContent = "Đang gửi...";
    try {
      const data = await api("/api/submit", {
        method: "POST",
        body: JSON.stringify({ id }),
      });
      if (["correct", "wrong", "pending"].includes(data.verdict)) kisSubmitted.ids.add(id);
      if (data.verdict === "correct") kisSubmitted.correct = true;
      const next = { correct: "Team đã có điểm cho task này, không cần nộp thêm.",
        wrong: "Bị trừ điểm phạt; có thể tìm và nộp ảnh khác.", pending: "Đang chờ trọng tài chấm, không cần làm gì." }[data.verdict];
      submitResult.textContent = `Kết quả: ${verdictLabel(data.verdict)}${next ? ` · ${next}` : ""}${data.message ? ` (${data.message})` : ""}`;
      const thumb = grid.querySelector(`img[data-id="${cssEscape(id)}"]`);
      if (thumb) thumb.classList.add("submitted");
    } catch (err) {
      if (err.status === 401) {
        submitResult.textContent = "Phiên đã hết hạn, vui lòng đăng nhập lại.";
        setTimeout(() => {
          closeLightbox();
          showLoggedOut();
        }, 1500);
        return;
      }
      submitResult.textContent = `Lỗi: ${err.message}`;
    }
  });

  function verdictLabel(v) {
    return (
      { correct: "ĐÚNG", wrong: "SAI", pending: "ĐANG CHỜ CHẤM", submitted: "ĐÃ NỘP", rejected: "BỊ TỪ CHỐI", error: "LỖI" }[v] || v
    );
  }

  function cssEscape(s) {
    return s.replace(/["\\]/g, "\\$&");
  }

  // Header chip + hint under the submit button: which evaluation, which task, and why submitting is blocked.
  function renderTaskStatus(task, error) {
    const picker = $("run-picker");
    const status = $("task-status");
    const hint = $("submit-hint");
    if (!picker || !status) return;
    // The server already submits to DRES_EVALUATION_ID or the demo evaluation; adopt it.
    if (task?.evaluation_id && !selectedEvaluationId) {
      selectedEvaluationId = task.evaluation_id;
      if (qaEvaluation && !qaEvaluation.value) qaEvaluation.value = task.evaluation_id;
    }
    const run = selectedEvaluationId
      ? (selectedEvaluationId.length > 12 ? selectedEvaluationId.slice(0, 8) + "…" : selectedEvaluationId)
      : "";
    let text, state, reason = "";
    if (error) {
      text = run ? `Buổi thi ${run} · lỗi DRES` : "Không kết nối được DRES";
      state = "missing";
      reason = error;
    } else if (!selectedEvaluationId) {
      text = "Chưa chọn buổi thi";
      state = "missing";
      reason = "Chưa chọn buổi thi: bấm “Chưa chọn buổi thi” ở góc trên để nhập Run ID.";
    } else if (!task?.running) {
      text = `Buổi thi ${run} · chưa có task`;
      state = "idle";
      reason = "Buổi thi chưa có task đang chạy.";
    } else {
      const duration = task.duration_s ? ` · ${Math.round(task.duration_s / 60)} phút` : "";
      text = `${task.name || "Task"} · ${task.task_type || "?"}${duration}`;
      state = "running";
    }
    status.textContent = text;
    status.title = `${selectedEvaluationId ? "Run ID: " + selectedEvaluationId : "Chưa chọn buổi thi"} — bấm để đổi`;
    picker.dataset.state = state;
    if (hint) hint.textContent = reason;
  }

  async function pollTask() {
    if (localQaMode) {
      setSearchEnabled(true);
      return;
    }
    try {
      const task = await api("/api/task");
      taskRunning = task.running;
      currentTaskType = task.task_type;
      if (task.running) setQaTask(`${selectedEvaluationId || ""}/${task.name || ""}`);
      renderTaskStatus(task);
      if (selectedEvaluationId) {
        qaEvaluationStatus.textContent = task.running
          ? `Task: ${task.name || ""} · ${task.task_type || "Chưa rõ loại"}`
          : "Chưa có task đang chạy.";
      }
      updateQaSubmitAvailability();
      setSearchEnabled(true);
      updateKisSubmitAvailability(task);
    } catch (err) {
      if (err.status === 401) {
        showLoggedOut();
        return;
      }
      // DRES unreachable or no evaluation configured: fail soft, keep retrying.
      renderTaskStatus(null, err.message || "Không kết nối được DRES");
      submitBtn.disabled = true;
      taskRunning = false;
      currentTaskType = null;
      updateQaSubmitAvailability();
      setSearchEnabled(true);
    } finally {
      setTimeout(pollTask, TASK_POLL_MS);
    }
  }

  function setSearchEnabled(enabled) {
    queryInput.disabled = !enabled;
    searchForm.querySelector("button").disabled = !enabled || searchForm.hasAttribute("aria-busy");
    if (!enabled) {
      searchMeta.textContent = "Chưa có task đang chạy";
    }
  }

  $("apply-evaluation-id").onclick = async () => {
    const raw = $("evaluation-id-input").value.trim();
    const id = raw.replace(/[?#].*$/, "").replace(/\/$/, "").split("/").pop();
    if (!id) {
      $("evaluation-message").textContent = "Bạn hãy nhập Run ID hoặc đường dẫn Viewer.";
      return;
    }
    const button = $("apply-evaluation-id");
    button.disabled = true;
    submitBtn.disabled = true;
    try {
      await api(`/api/evaluation/${encodeURIComponent(id)}`, { method: "POST" });
      selectedEvaluationId = id;
      qaEvaluation.value = id;
      qaEvaluationStatus.textContent = "Đã chọn buổi thi. Đang kiểm tra task…";
      await pollTaskOnce();
      $("evaluation-message").textContent = `Đã chọn buổi thi: ${id}.`;
      $("run-picker").open = false;
    } catch (err) {
      $("evaluation-message").textContent = err.message;
    } finally {
      button.disabled = false;
    }
  };

  async function pollTaskOnce() {
    try {
      const task = await api("/api/task");
      taskRunning = task.running;
      currentTaskType = task.task_type;
      if (task.running) setQaTask(`${selectedEvaluationId || ""}/${task.name || ""}`);
      renderTaskStatus(task);
      updateKisSubmitAvailability(task);
      qaEvaluationStatus.textContent = task.running
        ? `Task: ${task.name || ""} · ${task.task_type || "Chưa rõ loại"}`
        : "Chưa có task đang chạy.";
    } catch (err) {
      taskRunning = false;
      currentTaskType = null;
      qaEvaluationStatus.textContent = err.message;
    }
    updateQaSubmitAvailability();
  }

  $("qa-apply-evaluation").addEventListener("click", async () => {
    const id = qaEvaluation.value.trim().replace(/[?#].*$/, "").replace(/\/$/, "").split("/").pop();
    if (!id) {
      qaEvaluationStatus.textContent = "Nhập Run ID hoặc đường dẫn Viewer.";
      return;
    }
    const button = $("qa-apply-evaluation");
    button.disabled = true;
    try {
      await api(`/api/evaluation/${encodeURIComponent(id)}`, { method: "POST" });
      selectedEvaluationId = id;
      qaEvaluation.value = id;
      $("evaluation-id-input").value = id;
      $("evaluation-message").textContent = `Đã chọn buổi thi: ${id}.`;
      await pollTaskOnce();
    } catch (err) {
      qaEvaluationStatus.textContent = err.message;
    } finally {
      button.disabled = false;
    }
  });

  qaSubmitText.addEventListener("input", updateQaSubmitAvailability);
  qaSubmitAnswer.addEventListener("click", async () => {
    const text = qaSubmitText.value.trim();
    if (!text || qaSubmitAnswer.disabled) return;
    const earlier = qaSubmitted.answers;
    const same = earlier.some((a) => a.text.toLowerCase() === text.toLowerCase());
    const warning = same ? "\n\nĐáp án này ĐÃ được nộp cho task này rồi."
      : earlier.length ? `\n\nTask này đã nộp ${earlier.length} đáp án; nộp thêm có thể bị tính là một lần sai.` : "";
    if (!window.confirm(`Nộp đáp án “${text}” cho buổi thi ${selectedEvaluationId}?${warning}\n\nGiáo viên chấm tay nên sẽ không có kết quả trả về.`)) return;
    qaSubmitAnswer.disabled = true;
    qaSubmitStatus.textContent = "Đang nộp…";
    try {
      const result = await api("/api/submit/text", {
        method: "POST", body: JSON.stringify({ text }),
      });
      if (["submitted", "pending", "correct", "wrong"].includes(result.verdict)) {
        // Manual grading: whatever DRES says, the only fact is that the answer was received.
        const time = new Date().toLocaleTimeString("vi-VN", { hour12: false });
        qaSubmitted.answers.push({ time, text });
        renderQaSubmitHistory();
        qaSubmitStatus.textContent = `Đã nộp lúc ${time}. Chờ giáo viên chấm tay (không có phản hồi đúng/sai).`;
      } else {
        qaSubmitStatus.textContent = `${verdictLabel(result.verdict)}${result.message ? ` · ${result.message}` : ""}`;
      }
    } catch (err) {
      qaSubmitStatus.textContent = err.message;
    } finally {
      await pollTaskOnce();
    }
  });

  window.addEventListener("pageshow", (event) => {
    // Back/forward navigation can restore the old document, including the chat.
    if (event.persisted) location.reload();
  });

  (async function init() {
    try {
      localStorage.removeItem("castle5000_search_v2");
      localStorage.removeItem("castle5000_last_search");
    } catch {
      /* Browser storage may be unavailable. */
    }
    try {
      const me = await api("/api/me");
      showLoggedIn(me.username, me.role, me.mode);
      pollTask();
    } catch {
      showLoggedOut();
    }
  })();
})();
