"""Exercise the real local app in installed Chrome; block external requests."""
import argparse
import json
import socket
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".runtime" / "browser-tools"))

from playwright.sync_api import sync_playwright, expect


def check_no_index(context, output):
    import uvicorn
    from castle.app import create_app
    from castle.config import load_config

    with tempfile.TemporaryDirectory() as tmp, socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        cfg = replace(load_config(), artifacts_dir=Path(tmp) / "missing-index")
        server = uvicorn.Server(uvicorn.Config(create_app(cfg), host="127.0.0.1", port=port,
                                              log_level="error", access_log=False))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            assert server.started, "No-index test server did not start"
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}")
            expect(page.locator("#readyLabel")).to_have_text("Chưa có index")
            expect(page.locator("#notice")).to_contain_text("Chưa có index hợp lệ")
            page.locator("#query").fill("people in a kitchen")
            page.locator("#searchButton").click()
            expect(page.locator("#searchButton")).to_be_enabled()
            expect(page.locator("#notice")).to_contain_text("Chua co index hop le")
            page.screenshot(path=str(output / "noindex.png"))
            assert not errors, errors
            page.close()
        finally:
            server.should_exit = True
            thread.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    assert urlparse(args.url).hostname in ("127.0.0.1", "localhost")
    output = ROOT / "artifacts" / "browser"
    output.mkdir(parents=True, exist_ok=True)
    errors, external, checks = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})

        def local_only(route):
            if urlparse(route.request.url).hostname not in ("127.0.0.1", "localhost"):
                external.append(route.request.url)
                route.abort()
            else:
                route.continue_()

        context.route("**/*", local_only)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.locator("#readyLabel")).to_have_text("Index sẵn sàng")
        expect(page.locator(".card")).to_have_count(60)
        if args.expected_count is not None:
            expected = f"{args.expected_count:,}".replace(",", ".")
            expect(page.locator("#imageCount")).to_have_text(expected)
        page.wait_for_function("Array.from(document.querySelectorAll('.card img')).slice(0,8).every(im => im.naturalWidth > 0)")
        page.screenshot(path=str(output / "desktop.png"))
        checks.append("ready, corpus count, 60 initial cards, desktop screenshot")

        page.locator("#query").fill("a man in a blue shirt holding a kitchen towel")
        page.locator("#searchButton").click()
        expect(page.locator("#searchButton")).to_be_enabled(timeout=180000)
        expect(page.locator(".score").first).to_be_visible()
        checks.append("real text search")

        page.locator("#batch").select_option("Batch2")
        expect(page.locator("#searchButton")).to_be_enabled(timeout=30000)
        assert all("Batch2" in value for value in page.locator(".card small").all_text_contents())
        page.locator("#hour").select_option("15")
        expect(page.locator("#searchButton")).to_be_enabled(timeout=30000)
        assert all("15h" in value for value in page.locator(".card small").all_text_contents())
        checks.append("batch and hour filters")

        page.locator(".card").first.click()
        expect(page.locator("#imageDialog")).to_be_visible()
        expect(page.locator(".neighbor").first).to_be_visible()
        page.wait_for_function("document.getElementById('largeImage').naturalWidth === 1280")
        page.screenshot(path=str(output / "dialog.png"))
        original = page.locator("#selectedFilename").inner_text()
        page.locator(".neighbor:not(.selected)").first.click()
        expect(page.locator("#selectedFilename")).not_to_have_text(original)
        page.locator("#similarButton").click()
        expect(page.locator("#searchButton")).to_be_enabled(timeout=30000)
        expect(page.locator("#resultsTitle")).to_have_text("Ảnh tương tự")
        checks.append("full image, thumbnails, neighbors, image-to-image")

        page.locator("#hour").select_option("0")
        expect(page.locator("#searchButton")).to_be_enabled(timeout=30000)
        expect(page.locator("#empty")).to_be_visible()
        page.locator("#hour").select_option("")
        expect(page.locator("#searchButton")).to_be_enabled(timeout=30000)

        def fail_once(route):
            route.fulfill(status=503, content_type="application/json", body=json.dumps({"detail": "smoke: simulated model error"}))

        page.route("**/api/search", fail_once, times=1)
        page.locator("#searchButton").click()
        expect(page.locator("#notice")).to_contain_text("simulated model error")
        expect(page.locator("#searchButton")).to_be_enabled()
        page.locator("#searchButton").click()
        expect(page.locator("#searchButton")).to_be_enabled(timeout=30000)
        expect(page.locator("#notice")).to_be_hidden()
        checks.append("empty state and recovery after injected model error")

        page.locator("#connect").click()
        expect(page.locator("#evaluationControls")).to_be_visible()
        expect(page.locator("#evaluation option")).to_have_count(3)
        page.locator("#evaluation").select_option("local-practice")
        expect(page.locator("#taskLabel")).not_to_be_empty()
        expect(page.locator("#autoForm")).to_be_hidden()
        page.locator("#toggleAuto").click()
        expect(page.locator("#autoForm")).to_be_visible()
        page.locator("#toggleAuto").click()
        expect(page.locator("#autoForm")).to_be_hidden()
        expect(page.locator("#history")).to_contain_text("Chưa có lượt nộp.")
        checks.append("automatic submission defaults off, toggles on/off without sending")
        page.locator(".card").first.click()
        expect(page.locator("#submitButton")).to_be_enabled()
        page.once("dialog", lambda dialog: dialog.dismiss())
        page.locator("#submitButton").click()
        expect(page.locator("#submissionResult")).to_be_empty()
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator("#submitButton").click()
        expect(page.locator("#submissionResult")).to_contain_text("MOCK_ACCEPTED")
        page.locator("#closeDialog").click()
        page.locator("details summary").click()
        expect(page.locator("#history")).to_contain_text("MOCK_ACCEPTED")
        page.reload()
        expect(page.locator("#connectionLabel")).to_contain_text("Luyện local")
        expect(page.locator("#evaluation")).to_have_value("local-practice")
        checks.append("mock login, evaluation, cancel/confirm, history, refresh persistence")

        page.locator("#evaluation").select_option("local-practice-qa")
        expect(page.locator("#qaForm")).to_be_visible()
        expect(page.locator("#submitQa")).to_be_disabled()
        page.locator(".card").first.click()
        expect(page.locator("#submitButton")).to_be_disabled()
        page.locator("#closeDialog").click()
        page.locator("#qaAnswer").fill("  đỏ  ")
        page.once("dialog", lambda dialog: dialog.dismiss())
        page.locator("#submitQa").click()
        expect(page.locator("#qaResult")).to_be_empty()
        expect(page.locator("#qaAnswer")).to_have_value("  đỏ  ")
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator("#submitQa").click()
        expect(page.locator("#qaResult")).to_contain_text("MOCK_ACCEPTED")
        expect(page.locator("#qaAnswer")).to_be_empty()
        page.reload()
        expect(page.locator("#qaForm")).to_be_visible()
        page.locator("details summary").click()
        expect(page.locator("#history")).to_contain_text("QA · đỏ · MOCK_ACCEPTED")
        page.locator("#evaluation").select_option("local-practice")
        expect(page.locator("#qaForm")).to_be_hidden()
        checks.append("QA mock: correct task, image blocked, cancel/confirm, trimmed answer and history")

        context.set_offline(True)
        page.locator("#searchButton").click()
        expect(page.locator("#searchButton")).to_be_enabled()
        expect(page.locator("#notice")).to_be_visible()
        context.set_offline(False)
        page.locator("#searchButton").click()
        expect(page.locator("#searchButton")).to_be_enabled(timeout=30000)
        expect(page.locator("#notice")).to_be_hidden()
        checks.append("network failure and recovery")

        page.locator("#logout").click()
        expect(page.locator("#loginForm")).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "mobile.png"))
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        checks.append("logout and mobile layout without horizontal overflow")
        check_no_index(context, output)
        checks.append("real browser on isolated app without index: UI loads, search reports error")
        assert not errors, errors
        assert not external, external
        report = {"checks": checks, "page_errors": errors, "external_requests": external,
                  "browser": browser.version, "notice": "Real CLIP search; submission is local MOCK_ACCEPTED only. Error paths are explicitly injected."}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
