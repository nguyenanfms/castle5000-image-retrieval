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
  const taskName = $("task-name");
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
  const qaContextDot = $("qa-context-dot");
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

  const STORAGE_KEY = searchV2 ? "castle5000_search_v2" : "castle5000_last_search";
  const regionControls = $("search-v2-controls");
  const planEditor = $("search-plan-editor");
  const planStatus = $("search-plan-status");
  const planToggle = $("search-manual");
  const searchLanguage = $("search-language");
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
    searchForm.querySelector('button[type="submit"]').textContent = "Tìm kiếm";
    $("inspector-close").hidden = false;
  }
  const TASK_POLL_MS = 5000;

  let currentImageId = null;
  let currentResults = [];
  let taskRunning = false;
  let selectedEvaluationId = "";
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
      if (tabRetrievalQaBtn) tabRetrievalQaBtn.hidden = true;
    }
  }

  function showLoggedIn(username, role) {
    localQaMode = role === "LOCAL_QA";
    loginView.hidden = true;
    searchView.hidden = false;
    statusbar.hidden = false;
    userName.textContent = username;
    logoutBtn.hidden = localQaMode;
    submitBtn.hidden = localQaMode;
    document.querySelectorAll(".competition-settings").forEach((el) => { el.hidden = localQaMode; });
    if (localQaMode) taskName.textContent = "Thử QA · Gemini";
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
  }

  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    loginError.hidden = true;
    const username = $("username").value;
    const password = $("password").value;
    try {
      const data = await api("/api/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
      showLoggedIn(data.username, data.role);
      restoreLastSearch();
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
    $("search-results-heading").hidden = !searchV2 || !results.length || grid.hidden;
    $("search-results-count").textContent = `${results.length} ảnh`;
    grid.innerHTML = "";
    for (const r of results) {
      const tile = document.createElement(searchV2 ? "button" : "div");
      tile.className = "thumb";

      const img = document.createElement("img");
      img.src = r.thumb;
      img.loading = "lazy";
      img.dataset.id = r.id;
      const scoreText = !searchV2 && typeof r.score === "number" ? ` (score ${r.score.toFixed(3)})` : "";
      img.title = `${r.id}${scoreText}`;
      if (searchV2) {
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
      grid.appendChild(tile);
    }
  }

  function saveLastSearch(query, results) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ query, results }));
    } catch {
      /* storage may be unavailable; non-critical */
    }
  }

  function restoreLastSearch() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const { query, results } = JSON.parse(raw);
      queryInput.value = query || "";
      if (searchV2) return;
      if (Array.isArray(results) && results.length) {
        renderResults(results);
        searchMeta.textContent = `${results.length} kết quả (đã lưu từ lần tìm trước)`;
      }
    } catch {
      /* ignore corrupt/missing storage */
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
      saveLastSearch(query, data.results);
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
    if (phase === "planning" || phase === "plan-error") selectWorkflowStep("query");
    if (["review", "searching"].includes(phase)) selectWorkflowStep("conditions");
    if (phase === "search-error" || (phase === "done" && !workflowStep)) selectWorkflowStep("results");
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
    for (const node of $("search-flow").querySelectorAll(".flow-node")) {
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
        : "Không tìm thấy ảnh phù hợp.") : workflowPhase === "searching" ? "Đang tìm và đối chiếu các mục đã chọn."
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
        try {
          const plan = await api("/api/search/plan", {
            method: "POST", body: JSON.stringify({ query, language }),
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
          setPlan(language === "en" ? query : "");
          renderWorkflow("plan-error");
          showPlanEditor(true);
          showPlanStatus(err.message);
          $("search-global").focus();
          return;
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
      const response = await api("/api/search", {
        method: "POST",
        body: JSON.stringify({ query, detail_queries: detailQueries, conditions, mode: "clip_region", k: 24 }),
      });
      renderResults(response.results);
      renderWorkflow("done", response);
      searchMeta.textContent = response.notice || (response.results.length ? "" : "Không tìm thấy ảnh phù hợp.");
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ query: queryInput.value })); } catch {}
    } catch (err) {
      if (err.status === 401) showLoggedOut();
      else { renderWorkflow("search-error"); searchMeta.textContent = err.message; }
    } finally {
      searchInProgress = false;
      searchForm.removeAttribute("aria-busy");
      button.disabled = apply.disabled = false;
      planToggle.disabled = false;
      renderWorkflowPopup();
      if (action === "edited" && planEditor.hidden) $("flow-conditions").querySelector("button").focus();
      if (action === "selected" && workflowStep === "results") $("flow-results").querySelector("button").focus();
    }
  }

  if (searchV2) {
    planToggle.addEventListener("click", () => {
      if (!planEditor.hidden) {
        showPlanEditor(false);
        return;
      }
      if (currentPlanKey() !== planKey && !preparedPlan) {
        setPlan(searchLanguage.value === "en" ? queryInput.value.trim() : "");
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

  function openLightbox(id, prefillQuestion = null) {
    const imageChanged = currentImageId !== id;
    currentImageId = id;
    lightboxImg.src = `/images/${id}`;
    lightboxImg.alt = id;
    submitResult.textContent = "";
    lightboxId.textContent = id;
    if (imageChanged || prefillQuestion !== null) {
      resetLightboxQaPanel(prefillQuestion);
    }
    const result = currentResults.find((row) => row.id === id);
    if (searchV2) {
      selectedSearchImage = id;
      const cards = grid.querySelectorAll(".thumb");
      cards.forEach((card) => card.setAttribute("aria-pressed", String(card.dataset.id === id)));
      renderDemoEvidence(result);
    } else {
      renderMatchedRegions(result?.matches || []);
      lightboxImg.onload = () => {
        renderMatchedRegions(currentResults.find((row) => row.id === currentImageId)?.matches || []);
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
    detector_unavailable: "Chưa có dữ liệu phát hiện riêng cho đối tượng này.",
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

  function renderDemoEvidence(result) {
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
    function draw() {
      if (lightboxImg.naturalWidth && lightboxImg.naturalHeight) {
        visual.style.setProperty("--photo-ratio", lightboxImg.naturalWidth / lightboxImg.naturalHeight);
      }
      overlay.replaceChildren();
      overlay.setAttribute("hidden", "");
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
        selected = check;
        for (const chip of phrases.children) chip.setAttribute("aria-pressed", String(chip === button));
        const lowerBound = check.observed_count > 0 ? `Phát hiện ít nhất ${check.observed_count}` : "";
        detail.textContent = [conditionReason[check.reason], lowerBound].filter(Boolean).join(" · ");
        draw();
      });
      phrases.append(button);
    }
    panel.append(phrases, detail);
    phrases.children[Math.max(0, checks.findIndex((check) => check.evidence.some((point) => point.box)))]?.click();
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
    if (!Array.isArray(box) || box.length !== 4) return;
    const overlay = $("matched-overlay");
    const { w: natW, h: natH } = getNaturalOverlaySize();
    const ns = "http://www.w3.org/2000/svg";
    const [x1, y1, x2, y2] = box;
    overlay.replaceChildren();
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", String(x1 * natW));
    rect.setAttribute("y", String(y1 * natH));
    rect.setAttribute("width", String((x2 - x1) * natW));
    rect.setAttribute("height", String((y2 - y1) * natH));
    rect.setAttribute("fill", "none");
    rect.setAttribute("stroke", "#4f8cff");
    rect.setAttribute("stroke-width", "4");
    overlay.appendChild(rect);
    if (labelText) {
      const txt = document.createElementNS(ns, "text");
      txt.setAttribute("x", String(Math.max(4, x1 * natW + 6)));
      txt.setAttribute("y", String(Math.max(18, y1 * natH - 6)));
      txt.setAttribute("fill", "#4f8cff");
      txt.setAttribute("font-size", "16");
      txt.setAttribute("font-family", "sans-serif");
      txt.textContent = labelText;
      overlay.appendChild(txt);
    }
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

  function appendContextDivider(text) {
    if (!qaChatHistory) return;
    if (qaChatEmpty) qaChatEmpty.hidden = true;
    const div = document.createElement("div");
    div.className = "qa-context-divider";
    div.textContent = text;
    qaChatHistory.appendChild(div);
    scrollChatToBottom();
  }

  function setActiveQaImage(newImageId, notifyInChat = true) {
    const normalized = newImageId ? String(newImageId).trim() : null;
    const changed = activeQaImageId !== normalized;
    activeQaImageId = normalized;

    if (qaContextDot) {
      qaContextDot.classList.toggle("pinned", Boolean(activeQaImageId));
    }
    if (qaContextLabel) {
      qaContextLabel.textContent = activeQaImageId
        ? `Đang hỏi trên ảnh: ${activeQaImageId}`
        : "Tự tìm ảnh liên quan";
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

    if (changed && notifyInChat) {
      if (activeQaImageId) {
        appendContextDivider(`— Đang xét ảnh: ${activeQaImageId} (các câu hỏi tiếp theo sẽ hỏi trên cùng ảnh này) —`);
      } else {
        appendContextDivider("— Đã đổi ngữ cảnh: Tự tìm ảnh mới theo câu hỏi —");
      }
    }
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
      setActiveQaImage(null, true);
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
        setActiveQaImage(targetId, true);
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
      openLightbox(ev.image_id);
      if (Array.isArray(ev.box)) {
        highlightEvidenceBox(ev.box, ev.image_id);
      }
    });
    return btn;
  }

  function renderRetrievalPanel(turn, data = {}) {
    if (turn !== activeRetrievalTurn) return;
    const scope = data.examined_scope || {};
    const examined = new Set(scope.image_ids || data.examined_ids || []);
    const evidence = new Map((data.evidence || []).map(ev => [ev.image_id, ev]));
    const ids = [...new Set([...(scope.retrieved_image_ids || data.image_ids || []), ...examined, ...evidence.keys()])];
    const confirmed = data.status === "answered";
    $("qa-retrieval-question").textContent = turn?.question || "";
    $("qa-retrieval-question").hidden = !turn;
    $("qa-retrieval-summary").textContent = ids.length
      ? `${ids.length} ảnh tìm được · ${examined.size} ảnh đã đọc${confirmed ? ` · ${evidence.size} ảnh bằng chứng` : ""}`
      : turn ? (data.status === "error" ? "Truy xuất chưa hoàn tất. Bạn có thể thử lại." : data.status ? "Chưa có ảnh phù hợp cho câu hỏi này." : "Đang tìm ảnh cho câu hỏi…") : "Ảnh liên quan sẽ xuất hiện khi bạn đặt câu hỏi.";
    $("qa-retrieval-empty").hidden = ids.length > 0;
    const grid = $("qa-retrieval-grid");
    // Avoid rebuilding loaded thumbnails on every progress tick.
    const key = JSON.stringify([ids, [...examined], [...evidence], confirmed]);
    if (grid.dataset.contentKey === key) return;
    grid.dataset.contentKey = key;
    grid.replaceChildren();
    ids.forEach((id, index) => {
      const ev = evidence.get(id);
      const figure = document.createElement("figure");
      figure.className = "qa-retrieval-card" + (ev && confirmed ? " is-evidence" : "");
      const button = createEvidenceActionBtn(ev || { image_id: id }, "");
      button.setAttribute("aria-label", `Xem ảnh ${id}`);
      const img = document.createElement("img");
      img.src = "/thumbs/" + encodeURIComponent(id);
      img.alt = ev?.caption || id;
      img.loading = "lazy";
      img.addEventListener("error", () => { img.src = "/images/" + encodeURIComponent(id); }, { once: true });
      button.appendChild(img);
      const caption = document.createElement("figcaption");
      const name = document.createElement("span"); name.textContent = `${index + 1}. ${id}`;
      const badge = document.createElement("span"); badge.className = "qa-image-state";
      badge.textContent = ev ? (confirmed ? "Bằng chứng" : "Liên quan") : examined.has(id) ? "Đã đọc" : "Tìm được";
      caption.append(name, badge);
      figure.append(button, caption);
      if (ev?.caption) figure.title = ev.caption;
      grid.appendChild(figure);
    });
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
    reply.append(botLine, altContainer, footer);
    turnEl.appendChild(reply);

    if (qaChatHistory) {
      qaChatHistory.appendChild(turnEl);
      scrollChatToBottom();
    }

    const turn = { turnEl, botLine, botText, altContainer, footer, question, retrievalData: {} };
    turnEl.addEventListener("click", () => selectRetrievalTurn(turn));
    selectRetrievalTurn(turn);
    return turn;
  }

  function finalizeChatTurn(turn, resp, lang, fallbackImageId = null) {
    turn.botLine.classList.remove("waiting", "error");
    const shortDefaultInsufficient =
      lang === "en" ? "Image is not clear enough to determine." : "Ảnh chưa đủ rõ để xác định.";
    const displayText =
      resp.status === "answered"
        ? resp.answer || shortDefaultInsufficient
        : resp.message || shortDefaultInsufficient;
    turn.botText.textContent = displayText;
    turn.footer.hidden = false;
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "qa-copy-btn";
    copy.textContent = "Sao chép";
    copy.setAttribute("aria-label", "Sao chép câu trả lời");
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(displayText);
        copy.textContent = "Đã sao chép";
        setTimeout(() => { copy.textContent = "Sao chép"; }, 1800);
      } catch (_) { copy.textContent = "Chưa sao chép được"; }
    });
    turn.footer.appendChild(copy);

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
      sources.setAttribute("aria-label", "Ảnh đi kèm câu trả lời");
      const heading = document.createElement("p");
      heading.className = "qa-sources-title";
      heading.textContent = resp.status === "answered"
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
    const viewImages = document.createElement("button");
    viewImages.type = "button";
    viewImages.className = "qa-copy-btn";
    viewImages.textContent = "Xem ảnh truy xuất →";
    viewImages.addEventListener("click", () => {
      selectRetrievalTurn(turn);
      if (window.innerWidth <= 760) $("qa-retrieval-grid").scrollIntoView({ behavior: "smooth", block: "start" });
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
          setActiveQaImage(altEv.image_id, true);
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
            setActiveQaImage(primaryEv.image_id, true);
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
      setActiveQaImage(capturedImageId, true);
      const chatTurn = appendChatTurn(
        question,
        lang === "en" ? "Checking image…" : "Đang trả lời…"
      );

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

  function abortRetrievalQa() {
    retrievalQaSeq += 1;
    if (retrievalQaAbort) {
      retrievalQaAbort.abort();
      retrievalQaAbort = null;
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
      retrievalQaQuestion.value = button.dataset.qaSuggestion;
      resizeQaInput();
      retrievalQaQuestion.focus();
    });
  });

  $("qa-new-chat")?.addEventListener("click", () => {
    abortRetrievalQa();
    qaConversationId = crypto.randomUUID();
    setActiveQaImage(null, false);
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

      retrievalQaSeq += 1;
      const reqSeq = retrievalQaSeq;
      if (retrievalQaAbort) {
        retrievalQaAbort.abort();
      }
      const controller = new AbortController();
      retrievalQaAbort = controller;
      retrievalQaSubmit.disabled = true;

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
          const labels = lang === "en"
            ? { planning: "Understanding question…", retrieving: "Finding images…", observing: "Examining evidence…", synthesizing: "Combining evidence…", verifying: "Checking answer…" }
            : { planning: "Đang hiểu câu hỏi…", retrieving: "Đang tìm ảnh…", observing: "Đang đối chiếu…", synthesizing: "Đang tổng hợp…", verifying: "Đang kiểm tra câu trả lời…" };
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

  async function pollTask() {
    if (localQaMode) {
      setSearchEnabled(true);
      return;
    }
    try {
      const task = await api("/api/task");
      taskRunning = task.running;
      if (!task.running) {
        taskName.textContent = "Chưa có task đang chạy";
      } else {
        taskName.textContent = `${task.name} (${task.task_group || ""})`;
      }
      setSearchEnabled(true);
      submitBtn.disabled = !task.running || !selectedEvaluationId;
    } catch (err) {
      if (err.status === 401) {
        showLoggedOut();
        return;
      }
      // DRES unreachable or no evaluation configured: fail soft, keep retrying.
      taskName.textContent = "Không lấy được thông tin task";
      submitBtn.disabled = true;
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
      $("evaluation-message").textContent = `Đã chọn buổi thi: ${id}. Chờ kiểm tra task đang chạy.`;
    } catch (err) {
      $("evaluation-message").textContent = err.message;
    } finally {
      button.disabled = false;
    }
  };

  (async function init() {
    try {
      const me = await api("/api/me");
      showLoggedIn(me.username, me.role);
      restoreLastSearch();
      pollTask();
    } catch {
      showLoggedOut();
    }
  })();
})();
