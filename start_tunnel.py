# -*- coding: utf-8 -*-
"""EduPredict AI — Internet orqali ulash (ngrok tunnel)"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyngrok import ngrok
import webbrowser

print("\n" + "="*55)
print("  EduPredict AI -- Internet Tunnel")
print("="*55)

try:
    print("\n[...] Tunnel ochilmoqda...")
    http_tunnel = ngrok.connect(8501, "http")
    public_url = http_tunnel.public_url

    print("\n" + "="*55)
    print("  [OK] TUNNEL MUVAFFAQIYATLI OCHILDI!")
    print("="*55)
    print(f"\n  PUBLIC URL:\n")
    print(f"  >>> {public_url} <<<")
    print(f"\n  Bu URLni boshqa noutbukda brauzerga kiriting.")
    print(f"  Internetga ulangan har qanday qurilmadan ishlaydi.")
    print("\n  Tugatish uchun: Ctrl+C")
    print("="*55 + "\n")

    webbrowser.open(public_url)

    try:
        ngrok.get_ngrok_process().proc.wait()
    except KeyboardInterrupt:
        print("\n[STOP] Tunnel yopildi.")
        ngrok.disconnect(public_url)
        ngrok.kill()

except Exception as e:
    err = str(e)
    if "authtoken" in err.lower() or "4018" in err or "4041" in err:
        print("\n[!] ngrok BEPUL TOKEN kerak!")
        print("\nQadamlar:")
        print("  1. https://dashboard.ngrok.com/signup -- bepul registratsiya")
        print("  2. Dashboard > 'Your Authtoken' -- tokeni nusxalab oling")
        print("  3. Quyidagi buyruqni ishga tushiring (tokeni o'rnating):")
        print("\n     python -m pyngrok config add-authtoken SIZNING_TOKEN\n")
        print("  4. Keyin: python start_tunnel.py")
    else:
        print(f"\n[!] Xato: {e}")
        print("Avval run.bat ni ishga tushiring, keyin start_tunnel.py ni qayta ishga tushiring.")
    input("\nEnter bosing...")
    sys.exit(1)
