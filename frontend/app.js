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

  function updateQaSubmitAvailability() {
    const type = (currentTaskType || "").toLowerCase();
    const isQa = type.includes("question") && type.includes("answer");
    qaSubmitAnswer.disabled = localQaMode || !selectedEvaluationId || !taskRunning || !isQa || !qaSubmitText.value.trim();
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
      }
      grid.appendChild(tile);
    }
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
    no_person_region: "Không có vùng nào trong ảnh giống người (chỉ mục vùng RegionCLIP).",
    person_region_detected: "Có vùng giống người rõ ràng trong ảnh (chỉ mục vùng RegionCLIP).",
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

  function createCosineSummary(member) {
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
    const reference = createEvidenceActionBtn({image_id: best.image_id,
      regions: [{box: best.box, label: best.label}]}, `${best.image_id} · ${best.label}`);
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

  function createDistinctGroupPanel(estimate, evidenceByImage = new Map(), originalEstimate = estimate) {
    const panel = document.createElement("section");
    panel.className = "qa-distinct-panel";
    const header = document.createElement("div");
    header.className = "qa-distinct-header";
    const title = document.createElement("strong");
    const subject = estimate.category === "person" || estimate.category === "people" ? "người" : (estimate.category || "đối tượng");
    title.textContent = `${estimate.estimated_count} nhóm ${subject} · t = ${Number(estimate.threshold).toFixed(2)}`;
    const tag = document.createElement("span");
    tag.textContent = estimate.preview && originalEstimate && estimate.threshold !== originalEstimate.threshold
      ? `Đã chỉnh từ t = ${Number(originalEstimate.threshold).toFixed(2)}` : "Chưa xác minh";
    header.append(title, tag);
    const summary = document.createElement("p");
    const coverage = estimate.requested_images ? `${estimate.examined_images}/${estimate.requested_images}` : `${estimate.examined_images}`;
    summary.textContent = `Đã đối chiếu ${coverage} ảnh · ${estimate.assigned_regions} vùng được so sánh. Mỗi thẻ là một nhóm: ô trái là vùng đại diện, ô phải là ảnh nguồn.`;
    const warning = document.createElement("p");
    warning.className = "qa-distinct-warning";
    const notes = [];
    if (estimate.region_source === "regionclip_proposals") notes.push("Vùng R là đề xuất RegionCLIP, có thể sai hoặc thiếu.");
    if (estimate.region_source === "grounding_dino_on_demand") notes.push("Khung D do Grounding DINO dò theo câu hỏi, có thể sai hoặc bỏ sót.");
    if ([...evidenceByImage.values()].some(item => item.unverified_proposals?.length))
      notes.push("Khung gợi ý chưa tính được xem riêng ở từng thẻ.");
    warning.textContent = notes.join(" ");
    warning.hidden = !notes.length;
    const grid = document.createElement("div");
    grid.className = "qa-distinct-grid";
    for (const [groupIndex, group] of (estimate.groups || []).entries()) {
      const member = group.members?.[0];
      if (!member || !Array.isArray(member.box)) continue;
      const card = document.createElement("article");
      card.className = "qa-distinct-card";
      card.hidden = groupIndex >= 6;
      const label = document.createElement("strong");
      label.textContent = `${group.group_id} · ${group.members.length} vùng`;
      const images = document.createElement("div");
      images.className = "qa-distinct-images";
      const evidence = {image_id: member.image_id, regions: [{box: member.box, label: member.label}]};
      const cropButton = createEvidenceActionBtn(evidence, "");
      cropButton.className = "qa-distinct-image";
      cropButton.setAttribute("aria-label", `Xem vùng ${member.label} trong ảnh ${member.image_id}`);
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", `Vùng ${member.label}`);
      const image = document.createElementNS("http://www.w3.org/2000/svg", "image");
      const source = "/images/" + encodeURIComponent(member.image_id);
      image.setAttribute("href", source);
      svg.appendChild(image);
      const dimensions = new Image();
      dimensions.onload = () => {
        const w = dimensions.naturalWidth, h = dimensions.naturalHeight;
        const [x1, y1, x2, y2] = member.box;
        image.setAttribute("width", String(w));
        image.setAttribute("height", String(h));
        svg.setAttribute("viewBox", `${x1*w} ${y1*h} ${(x2-x1)*w} ${(y2-y1)*h}`);
      };
      dimensions.src = source;
      cropButton.appendChild(svg);
      const allEvidence = evidenceByImage.get(member.image_id) || evidence;
      const sourceButton = createEvidenceActionBtn(allEvidence, "");
      sourceButton.className = "qa-distinct-image";
      sourceButton.setAttribute("aria-label", `Xem tất cả vùng đã phát hiện trong ảnh ${member.image_id}`);
      const thumbnail = document.createElement("img");
      thumbnail.src = "/thumbs/" + encodeURIComponent(member.image_id);
      thumbnail.alt = `Ảnh nguồn ${member.image_id}`;
      thumbnail.addEventListener("error", () => { thumbnail.src = source; }, {once: true});
      sourceButton.appendChild(thumbnail);
      images.append(cropButton, sourceButton);
      const caption = document.createElement("small");
      caption.textContent = `Vùng đối tượng · Ảnh nguồn ${member.image_id}`;
      card.append(label, images, caption);
      if (allEvidence.unverified_proposals?.length) {
        card.appendChild(createEvidenceActionBtn({image_id: member.image_id,
          regions: allEvidence.unverified_proposals}, "Xem khung gợi ý chưa tính"));
      }
      const comparisonDetails = document.createElement("details");
      comparisonDetails.className = "qa-card-comparison";
      const comparisonTitle = document.createElement("summary");
      comparisonTitle.textContent = "Chi tiết đối chiếu";
      comparisonDetails.append(comparisonTitle, createCosineSummary(member));
      card.appendChild(comparisonDetails);
      if (group.members.length > 1) {
        const details = document.createElement("details");
        const detailsTitle = document.createElement("summary");
        detailsTitle.textContent = "Các ảnh trong nhóm";
        details.appendChild(detailsTitle);
        for (const item of group.members.slice(1)) {
          const entry = document.createElement("div");
          entry.className = "qa-distinct-member";
          entry.appendChild(createEvidenceActionBtn({image_id: item.image_id,
            regions: [{box: item.box, label: item.label}]}, `${item.image_id} · ${item.label}`));
          entry.appendChild(createCosineSummary(item));
          details.appendChild(entry);
        }
        card.appendChild(details);
      }
      grid.appendChild(card);
    }
    panel.append(header, summary, warning, grid);
    if ((estimate.groups || []).length > 6) {
      const more = document.createElement("button");
      more.type = "button";
      more.className = "qa-copy-btn qa-distinct-more";
      more.textContent = `Xem thêm ${estimate.groups.length - 6} nhóm`;
      more.addEventListener("click", () => {
        grid.querySelectorAll(".qa-distinct-card").forEach(card => { card.hidden = false; });
        more.hidden = true;
      });
      panel.appendChild(more);
    }
    return panel;
  }

  function createCosineHistory(estimate) {
    const history = document.createElement("div");
    history.className = "qa-cosine-history";
    const steps = (estimate?.groups || []).flatMap(group => (group.members || [])
      .filter(member => member.comparison)
      .map(member => ({member, groupId: group.group_id})))
      .sort((a, b) => a.member.sequence - b.member.sequence);
    if (steps.length) {
      const heading = document.createElement("h3");
      heading.textContent = `${steps.length} bước đối chiếu`;
      const note = document.createElement("p");
      note.textContent = `${estimate.preview ? "Đã chỉnh ngưỡng" : "Kết quả gốc"} · t = ${Number(estimate.threshold).toFixed(2)} · ${estimate.estimated_count} nhóm. Cosine ≥ t có thể được gộp; các vùng cùng ảnh không gộp chung.`;
      history.append(heading, note);
      for (const {member, groupId} of steps) {
        const info = member.comparison;
        const comparisons = [...(info.comparisons || [])].sort((a, b) => b.similarity - a.similarity);
        const best = comparisons.find(item => item.eligible);
        const step = document.createElement("details");
        step.className = "qa-cosine-step";
        const title = document.createElement("summary");
        const decision = info.decision === "matched" ? `Gộp vào ${groupId}` : `Tạo ${groupId}`;
        title.textContent = `#${member.sequence} · ${member.image_id} · ${member.label} → ${decision}` +
          (best ? ` · cao nhất ${best.similarity.toFixed(4)} với ${best.image_id} · ${best.label}` : " · chưa có vùng đủ điều kiện so sánh");
        step.appendChild(title);
        const current = createEvidenceActionBtn({image_id: member.image_id,
          regions: [{box: member.box, label: member.label}]}, `Xem vùng đang xét: ${member.image_id} · ${member.label}`);
        step.appendChild(current);
        if (!comparisons.length) {
          const empty = document.createElement("p");
          empty.textContent = "Vùng đầu tiên, chưa có vùng nào trước đó để so sánh.";
          step.appendChild(empty);
        }
        for (const item of comparisons) {
          const row = document.createElement("div");
          row.className = "qa-cosine-row";
          const reference = createEvidenceActionBtn({image_id: item.image_id,
            regions: [{box: item.box, label: item.label}]}, `${item.image_id} · ${item.label}`);
          const result = document.createElement("span");
          const status = item.eligible ? (item.above_threshold ? "đạt ngưỡng" : "dưới ngưỡng") : "không gộp: nhóm đã có vùng cùng ảnh";
          result.textContent = `${item.group_id} · cosine ${item.similarity.toFixed(4)} · ${status}`;
          row.append(reference, result);
          step.appendChild(row);
        }
        history.appendChild(step);
      }
    } else {
      const empty = document.createElement("p");
      empty.className = "qa-retrieval-empty";
      empty.textContent = "Chưa có bước đối chiếu cho câu trả lời này.";
      history.appendChild(empty);
    }
    return history;
  }

  function selectQaSideTab(tab) {
    const comparison = tab === "comparison";
    $("qa-images-panel").hidden = comparison;
    $("qa-comparison-panel").hidden = !comparison;
    for (const [id, selected] of [["qa-images-tab", !comparison], ["qa-comparison-tab", comparison]]) {
      const button = $(id);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    }
  }

  $("qa-images-tab").addEventListener("click", () => selectQaSideTab("images"));
  $("qa-comparison-tab").addEventListener("click", () => selectQaSideTab("comparison"));
  for (const [id, next] of [["qa-images-tab", "comparison"], ["qa-comparison-tab", "images"]]) {
    $(id).addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      selectQaSideTab(next);
      $(next === "images" ? "qa-images-tab" : "qa-comparison-tab").focus();
    });
  }

  function renderRetrievalPanel(turn, data = {}) {
    if (turn !== activeRetrievalTurn) return;
    const comparisonPanel = $("qa-comparison-panel");
    const comparisonKey = JSON.stringify(data.distinct_estimate || null);
    if (comparisonPanel.dataset.contentKey !== comparisonKey) {
      comparisonPanel.dataset.contentKey = comparisonKey;
      comparisonPanel.replaceChildren(createCosineHistory(data.distinct_estimate));
    }
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
    $("qa-legend").hidden = !turn || !ids.length;
    const grid = $("qa-retrieval-grid");
    // Avoid rebuilding loaded thumbnails on every progress tick.
    const key = JSON.stringify([ids, [...examined], [...evidence], confirmed, pinnedPreview]);
    if (grid.dataset.contentKey === key) return;
    grid.dataset.contentKey = key;
    grid.replaceChildren();
    ids.forEach((id, index) => {
      const ev = evidence.get(id);
      const figure = document.createElement("figure");
      figure.hidden = index >= 8;
      const verified = ev && confirmed && ev.verification_status !== "unverified";
      figure.className = "qa-retrieval-card" + (verified ? " is-evidence" : "");
      const button = createEvidenceActionBtn(ev || { image_id: id }, "");
      button.setAttribute("aria-label", `Xem ảnh ${id}`);
      const img = document.createElement("img");
      img.src = "/thumbs/" + encodeURIComponent(id);
      img.alt = ev?.caption || id;
      img.loading = "lazy";
      img.addEventListener("error", () => { img.src = "/images/" + encodeURIComponent(id); }, { once: true });
      button.appendChild(img);
      const caption = document.createElement("figcaption");
      const name = document.createElement("span"); name.textContent = pinnedPreview ? id : `${index + 1}. ${id}`;
      const badge = document.createElement("span");
      const state = pinnedPreview ? "pinned" : verified ? "evidence" : (ev || examined.has(id)) ? "read" : "unread";
      badge.className = `qa-image-state state-${state}`;
      badge.textContent = { pinned: "Đang hỏi", evidence: "Bằng chứng", read: "Đã đọc", unread: "Chưa đọc" }[state];
      figure.dataset.state = state;
      caption.append(name, badge);
      figure.append(button, caption);
      if (ev?.caption) figure.title = ev.caption;
      grid.appendChild(figure);
    });
    if (ids.length > 8) {
      const more = document.createElement("button");
      more.type = "button";
      more.className = "qa-retrieval-more";
      more.textContent = `Xem thêm ${ids.length - 8} ảnh`;
      more.addEventListener("click", () => {
        grid.querySelectorAll("figure").forEach(figure => { figure.hidden = false; });
        more.remove();
      });
      grid.appendChild(more);
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
  const QA_STAGE_ALIAS = { scanning: "retrieving", drafting: "observing" };

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
      const list = document.createElement("dl");
      list.className = "qa-plan";
      const rows = [
        ["Câu hỏi hiểu là", plan.question],
        ["Yêu cầu", QA_OPERATIONS[plan.operation] || plan.operation],
        ["Đối tượng", plan.subject],
        ["Phạm vi", QA_SCOPE_NAMES[plan.answer_scope] || plan.answer_scope],
        ["Điều kiện", plan.known_conditions?.join("; ")],
        ["Cần xác định", plan.information_needed],
        ["Truy vấn ảnh", plan.queries?.join("; ")],
        ["Cần làm rõ", plan.clarification],
      ];
      for (const [term, value] of rows) {
        if (!value) continue;
        const dt = document.createElement("dt");
        dt.textContent = term;
        const dd = document.createElement("dd");
        dd.textContent = value;
        list.append(dt, dd);
      }
      turn.progressList.appendChild(list);
    }
    if (progress.observations?.length) {
      const item = document.createElement("div");
      const details = document.createElement("details");
      details.className = "qa-observation-details";
      details.open = observationsOpen;
      const title = document.createElement("summary");
      title.textContent = `Mô hình quan sát ${progress.observations.length} ảnh`;
      const note = document.createElement("p");
      note.textContent = "Đây là mô tả và các vùng mô hình đề xuất; nội dung có thể sai hoặc chưa đầy đủ.";
      details.append(title, note);
      for (const observation of progress.observations) {
        const row = document.createElement("div");
        row.className = "qa-observation-row";
        const label = document.createElement("strong");
        label.textContent = `${observation.image_id} · ${observation.context_match === "yes" ? "phù hợp" : observation.context_match === "no" ? "không phù hợp" : "chưa rõ"}`;
        const text = document.createElement("p");
        text.textContent = observation.observation || "Mô hình không cung cấp mô tả.";
        row.append(label, text);
        if (observation.uncertainty) {
          const uncertain = document.createElement("p");
          uncertain.textContent = `Chưa chắc chắn: ${observation.uncertainty}`;
          row.appendChild(uncertain);
        }
        if (observation.regions?.length) {
          const regions = document.createElement("p");
          const state = {complete: "đầy đủ theo mô hình", partial: "có thể chưa đầy đủ", unknown: "chưa xác định"};
          regions.textContent = `${observation.regions.length} vùng đề xuất · ${state[observation.count_visibility] || state.unknown}: `;
          for (const region of observation.regions) {
            regions.appendChild(createEvidenceActionBtn({image_id: observation.image_id, regions: [region]}, region.label));
          }
          row.appendChild(regions);
        }
        if (observation.checks?.length) {
          const checks = document.createElement("p");
          const state = {met: "đạt", violated: "không đạt", unclear: "chưa rõ"};
          checks.textContent = observation.checks.map(check => `Điều kiện ${check.condition_id}: ${state[check.status] || check.status}${check.detail ? ` (${check.detail})` : ""}`).join(" · ");
          row.appendChild(checks);
        }
        details.appendChild(row);
      }
      item.appendChild(details);
      turn.progressList.appendChild(item);
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
  function createThresholdControl(turn, resp, lang, onChange) {
    const original = resp.distinct_estimate;
    const wrap = document.createElement("div");
    wrap.className = "qa-threshold";
    const label = document.createElement("label");
    label.className = "qa-threshold-label";
    const title = document.createElement("span");
    title.textContent = "Ngưỡng cosine t";
    const value = document.createElement("output");
    value.textContent = original.threshold.toFixed(2);
    label.append(title, value);
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0.50";
    slider.max = "0.99";
    slider.step = "0.01";
    slider.value = String(original.threshold);
    slider.setAttribute("aria-label", "Ngưỡng cosine để ghép cùng một đối tượng");
    const scale = document.createElement("div");
    scale.className = "qa-threshold-scale";
    scale.innerHTML = "<span>Gộp nhiều hơn → ít nhóm</span><span>Tách nhiều hơn → nhiều nhóm</span>";
    const footer = document.createElement("div");
    footer.className = "qa-threshold-footer";
    const hint = document.createElement("small");
    const reset = document.createElement("button");
    reset.type = "button";
    reset.className = "qa-copy-btn";
    reset.textContent = `Về t gốc (${original.threshold.toFixed(2)})`;
    reset.hidden = true;
    footer.append(hint, reset);
    wrap.append(label, slider, scale, footer);
    const canRegroup = Boolean(regroupDistinctEstimate(original, original.threshold));
    if (!canRegroup) {
      slider.disabled = true;
      hint.textContent = "Không có dữ liệu cosine để ghép lại; hãy hỏi lại câu này.";
      return wrap;
    }
    hint.textContent = "Kéo để ghép lại các vùng đã phát hiện; không gọi lại mô hình.";
    let pending = 0;
    const apply = (threshold) => {
      value.textContent = threshold.toFixed(2);
      const changed = Math.abs(threshold - original.threshold) > 1e-9;
      reset.hidden = !changed;
      const estimate = changed ? regroupDistinctEstimate(original, threshold) : original;
      if (estimate) onChange(estimate, changed);
    };
    slider.addEventListener("input", () => {
      value.textContent = Number(slider.value).toFixed(2);
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(() => apply(Number(slider.value)));
    });
    reset.addEventListener("click", () => {
      slider.value = String(original.threshold);
      apply(original.threshold);
    });
    return wrap;
  }

  function finalizeChatTurn(turn, resp, lang, fallbackImageId = null) {
    turn.botLine.classList.remove("waiting", "error");
    finishQaLive(turn, "done");
    const shortDefaultInsufficient =
      lang === "en" ? "Image is not clear enough to determine." : "Ảnh chưa đủ rõ để xác định.";
    let displayText =
      resp.status === "answered"
        ? resp.answer || shortDefaultInsufficient
        : resp.message || shortDefaultInsufficient;
    // Text sent to DRES: short form for estimates, full answer otherwise.
    turn.submitText = resp.status === "answered" ? resp.answer : null;
    const estimate = resp.distinct_estimate?.groups?.length ? resp.distinct_estimate : null;
    if (estimate) {
      const composed = composeDistinctAnswer(estimate, resp, lang);
      displayText = composed.main;
      turn.submitText = composed.short;
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

    turn.retrievalData = resp;
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
    if (estimate) {
      const evidenceByImage = new Map((resp.evidence || []).map(ev => [ev.image_id, ev]));
      const box = document.createElement("section");
      box.className = "qa-estimate-box";
      const details = document.createElement("details");
      details.className = "qa-group-disclosure";
      const summary = document.createElement("summary");
      const setSummary = (current) => {
        summary.textContent = `Xem ${current.estimated_count} nhóm ${distinctNoun(current, lang)}`;
      };
      setSummary(estimate);
      let panel = createDistinctGroupPanel(estimate, evidenceByImage, estimate);
      details.append(summary, panel);
      const control = createThresholdControl(turn, resp, lang, (current) => {
        const composed = composeDistinctAnswer(current, resp, lang);
        turn.botText.textContent = composed.main;
        turn.displayText = composed.main;
        turn.submitText = composed.short;
        renderCaveats(turn, composed.caveats);
        renderAnswerMeta(turn, resp, current);
        setSummary(current);
        const next = createDistinctGroupPanel(current, evidenceByImage, estimate);
        panel.replaceWith(next);
        panel = next;
        turn.retrievalData = { ...turn.retrievalData, distinct_estimate: current };
        renderRetrievalPanel(turn, turn.retrievalData);
      });
      box.append(control, details);
      const sources = turn.footer.querySelector(".qa-answer-sources");
      if (sources) sources.after(box);
      else turn.footer.prepend(box);
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
      selectQaSideTab("images");
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
            ? { planning: "Understanding question…", scanning: "Scanning the image catalog…", retrieving: "Finding images…", observing: "Examining evidence…", synthesizing: "Combining evidence…", drafting: "Trying a draft answer…", verifying: "Checking answer…" }
            : { planning: "Đang hiểu câu hỏi…", scanning: "Đang rà danh mục ảnh…", retrieving: "Đang tìm ảnh…", observing: "Đang đối chiếu…", synthesizing: "Đang tổng hợp…", drafting: "Đang thử câu trả lời nháp…", verifying: "Đang kiểm tra câu trả lời…" };
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
    confirmDialog.querySelector("p").textContent = `Nộp ${currentImageId} vào buổi thi ${selectedEvaluationId}? Nộp sai có thể bị trừ điểm.`;
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
      submitResult.textContent = `Kết quả: ${verdictLabel(data.verdict)} (${data.message || ""})`;
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
      { correct: "ĐÚNG", wrong: "SAI", pending: "ĐANG CHỜ CHẤM", rejected: "BỊ TỪ CHỐI", error: "LỖI" }[v] || v
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
      renderTaskStatus(task);
      if (selectedEvaluationId) {
        qaEvaluationStatus.textContent = task.running
          ? `Task: ${task.name || ""} · ${task.task_type || "Chưa rõ loại"}`
          : "Chưa có task đang chạy.";
      }
      updateQaSubmitAvailability();
      setSearchEnabled(true);
      submitBtn.disabled = !task.running || !selectedEvaluationId;
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
      renderTaskStatus(task);
      submitBtn.disabled = !task.running || !selectedEvaluationId;
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
    if (!window.confirm(`Nộp đáp án “${text}” cho buổi thi ${selectedEvaluationId}?`)) return;
    qaSubmitAnswer.disabled = true;
    qaSubmitStatus.textContent = "Đang nộp…";
    try {
      const result = await api("/api/submit/text", {
        method: "POST", body: JSON.stringify({ text }),
      });
      if (result.verdict === "pending") {
        qaSubmitStatus.textContent = "Đã gửi đáp án. Giáo viên sẽ chấm thủ công.";
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
