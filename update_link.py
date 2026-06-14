import re, os, subprocess, time, webbrowser, shutil

BASE = os.path.dirname(os.path.abspath(__file__))
GIT = r"C:\Program Files\Git\cmd\git.exe"
LOG = os.path.join(BASE, "tunnel_log.txt")
HTML = os.path.join(BASE, "docs", "index.html")
PLACEHOLDER = "TUNNEL_URL_PLACEHOLDER"

def get_url(timeout=30):
    for _ in range(timeout):
        try:
            txt = open(LOG).read()
            m = re.search(r"https://[\w\-]+\.trycloudflare\.com", txt)
            if m:
                return m.group(0)
        except:
            pass
        time.sleep(1)
    return None

url = get_url()
if not url:
    print("URL topilmadi, qayta urinilmoqda...")
    time.sleep(10)
    url = get_url(20)

if url:
    # HTML faylni yangilash
    with open(HTML) as f:
        html = f.read()
    html = re.sub(r"https://[\w\-]+\.trycloudflare\.com", url, html)
    html = html.replace(PLACEHOLDER, url)
    with open(HTML, "w") as f:
        f.write(html)

    # GitHub Pages ga push
    try:
        env = os.environ.copy()
        env["PATH"] = r"C:\Program Files\Git\cmd;" + env.get("PATH", "")
        subprocess.run(["git", "-C", BASE, "add", "docs/index.html"], env=env, capture_output=True)
        subprocess.run(["git", "-C", BASE, "commit", "-m", f"Update tunnel URL: {url}"], env=env, capture_output=True)
        subprocess.run(["git", "-C", BASE, "push", "origin", "main"], env=env, capture_output=True)
        print(f"\n  GitHub Pages yangilandi!")
    except Exception as e:
        print(f"  Push xato: {e}")

    print("\n" + "="*60)
    print(f"  DOIMIY LINK (hech o'zgarmaydi):")
    print(f"  https://dinaraisrailova-ship-it.github.io/edupredict-ai/")
    print(f"\n  HOZIRGI LINK:")
    print(f"  {url}")
    print("="*60)

    webbrowser.open(url)
else:
    print("  Tunnel URL topilmadi!")
