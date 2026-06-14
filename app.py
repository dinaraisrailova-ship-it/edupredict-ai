import os, sys, json, warnings, traceback
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto",
)

BASE        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, "data", "raw", "students.csv")
MODELS_DIR  = os.path.join(BASE, "models")
FIGURES_DIR = os.path.join(BASE, "reports", "figures")
REPORT_PATH = os.path.join(BASE, "reports", "model_results.json")

# ══════════════════════════════════════ CSS + JS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@400;500;600;700;800;900&display=swap');

/* ═══════════════════════════════════════════
   NEON CYBER — Cyan × Violet × Deep Black
═══════════════════════════════════════════ */

html,body,[class*="css"]{font-family:'Inter',sans-serif;scroll-behavior:smooth;}
*{box-sizing:border-box;}

/* ── BACKGROUND: cyber grid ── */
.stApp{
  background:#010610;
  background-image:
    linear-gradient(rgba(0,212,255,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,212,255,.03) 1px,transparent 1px),
    radial-gradient(ellipse 80% 60% at 85% 5%,rgba(139,92,246,.18) 0%,transparent 55%),
    radial-gradient(ellipse 55% 45% at 6% 94%,rgba(0,212,255,.1) 0%,transparent 55%),
    radial-gradient(ellipse 70% 60% at 50% 50%,rgba(139,92,246,.06) 0%,transparent 65%);
  background-size:60px 60px,60px 60px,100% 100%,100% 100%,100% 100%;
  min-height:100vh;
}
.stApp::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.018) 3px,rgba(0,0,0,.018) 6px);
}
.stApp::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(circle 700px at 88% 4%,rgba(139,92,246,.16),transparent 55%),
    radial-gradient(circle 500px at 4% 96%,rgba(0,212,255,.1),transparent 55%),
    radial-gradient(circle 400px at 55% 80%,rgba(139,92,246,.06),transparent 50%);
  animation:orbPulse 14s ease-in-out infinite alternate;
}
@keyframes orbPulse{0%{opacity:.5;transform:scale(1)}100%{opacity:1;transform:scale(1.1)}}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#010814 0%,#010610 50%,#020812 100%)!important;
  border-right:1px solid rgba(0,212,255,.12)!important;
  box-shadow:4px 0 50px rgba(0,0,0,.7),0 0 1px rgba(0,212,255,.14),inset -1px 0 0 rgba(0,212,255,.06)!important;
  backdrop-filter:blur(20px)!important;
}
section[data-testid="stSidebar"] *{color:#94A3B8!important;}
section[data-testid="stSidebar"] .stRadio label{
  border-radius:10px!important;
  transition:all .22s cubic-bezier(.4,0,.2,1)!important;
  padding:11px 16px!important;margin:3px 0!important;
  border:1px solid transparent!important;
  cursor:pointer!important;font-size:.86rem!important;font-weight:600!important;
  font-family:'Space Grotesk',sans-serif!important;letter-spacing:.02em!important;
}
section[data-testid="stSidebar"] .stRadio label:hover{
  background:rgba(0,212,255,.08)!important;
  border-color:rgba(0,212,255,.22)!important;
  color:#00D4FF!important;transform:translateX(6px)!important;
  box-shadow:0 0 16px rgba(0,212,255,.1),inset 0 0 12px rgba(0,212,255,.04)!important;
}

/* ── TEXT ── */
.stMarkdown p,.stText,p,li{color:#94A3B8!important;line-height:1.8;}
h1,h2,h3,h4{color:#E2E8F0!important;font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.02em;font-weight:800!important;}
strong,b{color:#E2E8F0!important;}
code{
  font-family:'JetBrains Mono',monospace!important;
  background:rgba(0,212,255,.08)!important;border-radius:6px!important;
  padding:2px 9px!important;color:#00D4FF!important;font-size:.84em!important;
  border:1px solid rgba(0,212,255,.2)!important;
}

/* ── HERO ── */
.hero{
  position:relative;border-radius:20px;padding:56px 60px;margin-bottom:36px;
  overflow:hidden;
  background:linear-gradient(135deg,#010B14 0%,#020F1C 20%,#041424 40%,#030E1A 60%,#010910 80%,#020810 100%);
  border:1px solid rgba(0,212,255,.22);
  box-shadow:0 0 0 1px rgba(0,212,255,.06),0 0 60px rgba(0,212,255,.1),0 24px 80px rgba(0,0,0,.65),inset 0 1px 0 rgba(0,212,255,.14);
}
.hero::before{
  content:'';position:absolute;inset:0;border-radius:20px;
  background:linear-gradient(rgba(0,212,255,.022) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,.022) 1px,transparent 1px);
  background-size:40px 40px;
}
.hero::after{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,rgba(0,212,255,.6) 20%,rgba(139,92,246,.8) 50%,rgba(0,212,255,.6) 80%,transparent);
  background-size:200% auto;animation:heroShine 3s ease-in-out infinite;
}
@keyframes heroShine{0%{background-position:200%}100%{background-position:-200%}}
.hero-content{position:relative;z-index:1;}

.eyebrow{
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.28);
  border-radius:6px;padding:6px 20px;
  font-size:.68rem;font-weight:700;color:#00D4FF!important;
  letter-spacing:.16em;text-transform:uppercase;margin-bottom:20px;
  font-family:'JetBrains Mono',monospace;
  box-shadow:0 0 16px rgba(0,212,255,.15),inset 0 0 8px rgba(0,212,255,.05);
  animation:eyebrowPulse 3s ease-in-out infinite;
}
@keyframes eyebrowPulse{
  0%,100%{box-shadow:0 0 16px rgba(0,212,255,.15),inset 0 0 8px rgba(0,212,255,.05)}
  50%{box-shadow:0 0 30px rgba(0,212,255,.32),0 0 50px rgba(0,212,255,.1),inset 0 0 12px rgba(0,212,255,.08)}
}

.htitle{
  font-family:'Space Grotesk',sans-serif!important;
  font-size:3.2rem;font-weight:900;line-height:1.06;letter-spacing:-.03em;margin-bottom:16px;
  background:linear-gradient(135deg,#ffffff 0%,#00D4FF 38%,#8B5CF6 72%,#ffffff 100%);
  background-size:300% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:titleFlow 6s ease-in-out infinite;
}
@keyframes titleFlow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}

.hsub{font-size:1.05rem;color:rgba(148,163,184,.85)!important;line-height:1.8;max-width:580px;}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin-top:24px;}
.chip{
  background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.2);
  border-radius:6px;padding:6px 18px;
  font-size:.76rem;color:#00D4FF!important;font-weight:600;
  font-family:'JetBrains Mono',monospace;transition:all .2s;cursor:default;
}
.chip:hover{background:rgba(0,212,255,.14);border-color:rgba(0,212,255,.5);transform:translateY(-3px);box-shadow:0 4px 20px rgba(0,212,255,.15);}

/* ── STAT CARDS ── */
.stat{
  background:rgba(0,212,255,.02);border:1px solid rgba(0,212,255,.1);
  border-radius:16px;padding:30px 20px;text-align:center;
  position:relative;overflow:hidden;
  transition:all .3s cubic-bezier(.34,1.56,.64,1);
  box-shadow:0 4px 24px rgba(0,0,0,.35);cursor:default;
}
.stat::before{
  content:'';position:absolute;inset:0;border-radius:16px;opacity:0;
  background:radial-gradient(ellipse at 50% 0%,rgba(0,212,255,.1),transparent 65%);
  transition:opacity .3s;
}
.stat::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,#00D4FF,#8B5CF6,#00D4FF,transparent);
  background-size:400% auto;animation:flowGlow 3s linear infinite;
}
@keyframes flowGlow{0%{background-position:0%}100%{background-position:400%}}
.stat:hover{transform:translateY(-12px) scale(1.04);border-color:rgba(0,212,255,.36);
  box-shadow:0 24px 70px rgba(0,0,0,.5),0 0 40px rgba(0,212,255,.14),inset 0 0 20px rgba(0,212,255,.04);}
.stat:hover::before{opacity:1;}
.si{font-size:2.4rem;display:block;margin-bottom:14px;animation:iconBob 3s ease-in-out infinite;filter:drop-shadow(0 0 14px rgba(0,212,255,.55));}
@keyframes iconBob{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(-7px) scale(1.1)}}
.sv{font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:900;background:linear-gradient(135deg,#00D4FF,#8B5CF6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;}
.sl{font-size:.67rem;color:#64748B!important;font-weight:700;text-transform:uppercase;letter-spacing:.14em;margin-top:12px;font-family:'JetBrains Mono',monospace;}

/* ── METRIC ── */
div[data-testid="metric-container"]{
  background:rgba(0,212,255,.04)!important;border:1px solid rgba(0,212,255,.14)!important;
  border-radius:14px!important;padding:22px!important;box-shadow:0 4px 24px rgba(0,0,0,.3),0 0 1px rgba(0,212,255,.1)!important;
  transition:all .25s cubic-bezier(.4,0,.2,1)!important;position:relative!important;overflow:hidden!important;
  backdrop-filter:blur(8px)!important;
}
div[data-testid="metric-container"]::before{
  content:'';position:absolute;inset:0;border-radius:14px;pointer-events:none;
  background:linear-gradient(135deg,rgba(0,212,255,.03) 0%,transparent 60%);
}
div[data-testid="metric-container"]::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,#00D4FF,#8B5CF6,#00D4FF,transparent);
  background-size:400% auto;animation:flowGlow 3s linear infinite;
}
div[data-testid="metric-container"]:hover{border-color:rgba(0,212,255,.36)!important;box-shadow:0 12px 40px rgba(0,0,0,.35),0 0 30px rgba(0,212,255,.12)!important;transform:translateY(-5px)!important;}
div[data-testid="metric-container"] label{color:#94A3B8!important;font-size:.7rem!important;text-transform:uppercase;letter-spacing:.1em;font-weight:700!important;font-family:'JetBrains Mono',monospace!important;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{background:linear-gradient(135deg,#00D4FF,#8B5CF6)!important;-webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;font-weight:900!important;font-size:1.75rem!important;font-family:'Space Grotesk',sans-serif!important;}

/* ── PRIMARY BUTTON ── */
div.stButton>button[kind="primary"]{
  position:relative!important;overflow:hidden!important;
  background:linear-gradient(135deg,#0369A1 0%,#0891B2 30%,#06B6D4 60%,#22D3EE 85%,#67E8F9 100%)!important;
  background-size:250% auto!important;
  color:#020810!important;border:none!important;border-radius:10px!important;
  padding:16px 40px!important;font-size:.93rem!important;font-weight:800!important;
  font-family:'Space Grotesk',sans-serif!important;letter-spacing:.06em!important;text-transform:uppercase!important;
  box-shadow:0 0 24px rgba(6,182,212,.5),0 4px 16px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.2)!important;
  transition:all .28s cubic-bezier(.4,0,.2,1)!important;width:100%!important;
}
div.stButton>button[kind="primary"]::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent);transition:left .4s ease!important;}
div.stButton>button[kind="primary"]:hover::before{left:100%!important;}
div.stButton>button[kind="primary"]:hover{transform:translateY(-4px)!important;background-position:right center!important;box-shadow:0 0 40px rgba(6,182,212,.7),0 0 80px rgba(6,182,212,.2),0 8px 28px rgba(0,0,0,.3)!important;}
div.stButton>button[kind="primary"]:active{transform:translateY(0) scale(.97)!important;}

/* ── SECONDARY BUTTON ── */
div.stButton>button[kind="secondary"]{
  position:relative!important;overflow:hidden!important;
  background:rgba(0,212,255,.05)!important;color:#00D4FF!important;
  border:1px solid rgba(0,212,255,.24)!important;border-radius:10px!important;
  font-weight:700!important;letter-spacing:.03em!important;
  font-family:'Space Grotesk',sans-serif!important;transition:all .22s cubic-bezier(.4,0,.2,1)!important;
}
div.stButton>button[kind="secondary"]:hover{background:rgba(0,212,255,.12)!important;border-color:rgba(0,212,255,.52)!important;box-shadow:0 0 20px rgba(0,212,255,.2)!important;transform:translateY(-3px)!important;color:#67E8F9!important;}
div.stButton>button[kind="secondary"]:active{transform:translateY(0) scale(.97)!important;}

/* ── DOWNLOAD BUTTON ── */
div.stDownloadButton>button{background:rgba(255,255,255,.03)!important;color:#475569!important;border:1px solid rgba(255,255,255,.07)!important;border-radius:10px!important;font-weight:700!important;font-family:'Space Grotesk',sans-serif!important;width:100%!important;transition:all .22s!important;}
div.stDownloadButton>button:hover{background:rgba(0,212,255,.08)!important;color:#00D4FF!important;border-color:rgba(0,212,255,.28)!important;box-shadow:0 0 16px rgba(0,212,255,.12)!important;transform:translateY(-2px)!important;}

/* ── TABS ── */
div[data-baseweb="tab-list"]{background:rgba(0,212,255,.03)!important;border:1px solid rgba(0,212,255,.1)!important;border-radius:10px!important;padding:5px!important;gap:4px!important;border-bottom:none!important;}
div[data-baseweb="tab"]{border-radius:7px!important;color:#334155!important;font-weight:700!important;transition:all .2s!important;font-size:.85rem!important;font-family:'Space Grotesk',sans-serif!important;}
div[data-baseweb="tab"]:hover{color:#00D4FF!important;background:rgba(0,212,255,.08)!important;}
div[aria-selected="true"][data-baseweb="tab"]{background:linear-gradient(135deg,#0369A1,#0891B2,#06B6D4)!important;color:#fff!important;font-weight:800!important;box-shadow:0 0 16px rgba(6,182,212,.4)!important;}

/* ── INPUTS ── */
div[data-baseweb="select"]>div{background:rgba(0,212,255,.03)!important;border:1px solid rgba(0,212,255,.1)!important;border-radius:10px!important;color:#E2E8F0!important;transition:all .2s!important;}
div[data-baseweb="select"]>div:focus-within{border-color:rgba(0,212,255,.48)!important;box-shadow:0 0 0 3px rgba(0,212,255,.1),0 0 16px rgba(0,212,255,.1)!important;}
input[type="number"],input[type="text"]{background:rgba(0,212,255,.03)!important;border:1px solid rgba(0,212,255,.1)!important;border-radius:10px!important;color:#E2E8F0!important;transition:all .2s!important;}
input[type="number"]:focus,input[type="text"]:focus{border-color:rgba(0,212,255,.48)!important;box-shadow:0 0 0 3px rgba(0,212,255,.1),0 0 16px rgba(0,212,255,.08)!important;outline:none!important;}

/* ── NUMBER BUTTONS ── */
div[data-testid="stNumberInput"] button{background:rgba(0,212,255,.07)!important;border-color:rgba(0,212,255,.18)!important;color:#00D4FF!important;transition:all .2s!important;border-radius:7px!important;font-weight:700!important;}
div[data-testid="stNumberInput"] button:hover{background:rgba(0,212,255,.18)!important;box-shadow:0 0 12px rgba(0,212,255,.2)!important;}

/* ── SLIDER ── */
div[data-testid="stSlider"]>div>div>div{background:linear-gradient(90deg,rgba(0,212,255,.3),rgba(139,92,246,.3))!important;height:3px!important;border-radius:3px!important;}
div[data-testid="stSlider"] div[role="slider"]{background:linear-gradient(135deg,#0891B2,#06B6D4,#22D3EE)!important;box-shadow:0 0 16px rgba(6,182,212,.7),0 0 4px rgba(0,212,255,1)!important;width:20px!important;height:20px!important;transition:transform .18s!important;border:2px solid rgba(255,255,255,.4)!important;}
div[data-testid="stSlider"] div[role="slider"]:hover{transform:scale(1.35)!important;}

/* ── MULTISELECT ── */
span[data-baseweb="tag"]{background:rgba(0,212,255,.1)!important;border:1px solid rgba(0,212,255,.24)!important;border-radius:6px!important;color:#00D4FF!important;font-weight:600!important;}

/* ── DATAFRAME ── */
div[data-testid="stDataFrame"]{border-radius:12px!important;overflow:hidden!important;border:1px solid rgba(0,212,255,.08)!important;box-shadow:0 4px 24px rgba(0,0,0,.3),0 0 20px rgba(0,212,255,.04)!important;}

/* ── ALERTS ── */
div[data-testid="stAlert"]{border-radius:12px!important;box-shadow:0 4px 16px rgba(0,0,0,.25)!important;border-left:3px solid #06B6D4!important;background:rgba(6,182,212,.06)!important;}

/* ── FILE UPLOADER ── */
section[data-testid="stFileUploaderDropzone"]{background:rgba(0,212,255,.03)!important;border:2px dashed rgba(0,212,255,.18)!important;border-radius:16px!important;transition:all .3s!important;}
section[data-testid="stFileUploaderDropzone"]:hover{background:rgba(0,212,255,.07)!important;border-color:rgba(0,212,255,.48)!important;box-shadow:0 0 30px rgba(0,212,255,.1)!important;}

/* ── RESULT BADGE ── */
.rbadge{
  display:block;text-align:center;border-radius:12px;padding:32px 56px;
  font-family:'Space Grotesk',sans-serif;font-size:1.85rem;font-weight:900;
  animation:popIn .52s cubic-bezier(.34,1.56,.64,1) both;
  letter-spacing:.05em;text-transform:uppercase;position:relative;overflow:hidden;
}
.rbadge::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.25),transparent);animation:shineSweep 2s ease-in-out .8s infinite;}
.rbadge::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:rgba(255,255,255,.5);}
@keyframes shineSweep{0%{left:-100%}100%{left:200%}}
@keyframes popIn{from{transform:scale(.4) translateY(24px);opacity:0}to{transform:scale(1) translateY(0);opacity:1}}
.rb-d{background:linear-gradient(135deg,#3B0000,#7F1D1D,#B91C1C,#EF4444);color:#fff!important;border:1px solid rgba(239,68,68,.38);box-shadow:0 0 40px rgba(239,68,68,.4),0 0 80px rgba(239,68,68,.1),0 12px 40px rgba(0,0,0,.5);}
.rb-e{background:linear-gradient(135deg,#2D1000,#78350F,#B45309,#F59E0B);color:#fff!important;border:1px solid rgba(245,158,11,.38);box-shadow:0 0 40px rgba(245,158,11,.35),0 0 80px rgba(245,158,11,.08),0 12px 40px rgba(0,0,0,.5);}
.rb-g{background:linear-gradient(135deg,#001A0E,#064E3B,#059669,#34D399);color:#fff!important;border:1px solid rgba(52,211,153,.38);box-shadow:0 0 40px rgba(52,211,153,.35),0 0 80px rgba(52,211,153,.08),0 12px 40px rgba(0,0,0,.5);}

/* ── PROBABILITY BARS ── */
.pb{margin:14px 0;}
.pbh{display:flex;justify-content:space-between;margin-bottom:7px;font-size:.82rem;font-weight:700;color:#64748B!important;font-family:'JetBrains Mono',monospace;}
.pbb{height:26px;background:rgba(255,255,255,.04);border-radius:4px;overflow:hidden;border:1px solid rgba(255,255,255,.05);}
.pbf{height:100%;border-radius:4px;display:flex;align-items:center;justify-content:flex-end;padding-right:11px;font-size:.78rem;font-weight:800;color:#020810;animation:barFill .8s cubic-bezier(.4,0,.2,1) both;position:relative;overflow:hidden;font-family:'JetBrains Mono',monospace;}
.pbf::after{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.3),transparent);animation:shineSweep 2s ease-in-out .8s infinite;}
@keyframes barFill{from{width:0%!important}to{}}

/* ── LEADERBOARD ── */
.lb{display:flex;align-items:center;gap:16px;background:rgba(0,212,255,.02);border:1px solid rgba(0,212,255,.09);border-radius:12px;padding:14px 20px;margin-bottom:8px;transition:all .25s cubic-bezier(.4,0,.2,1);box-shadow:0 2px 12px rgba(0,0,0,.28);position:relative;overflow:hidden;}
.lb::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,#06B6D4,#8B5CF6);opacity:0;transition:opacity .2s;}
.lb:hover{background:rgba(0,212,255,.06);border-color:rgba(0,212,255,.24);transform:translateX(6px);box-shadow:0 0 20px rgba(0,212,255,.08);}
.lb:hover::before{opacity:1;}
.lb.gold{border-color:rgba(251,191,36,.28);background:rgba(251,191,36,.04);}
.lb.silver{border-color:rgba(148,163,184,.18);background:rgba(148,163,184,.03);}
.lb.bronze{border-color:rgba(180,83,9,.22);background:rgba(180,83,9,.03);}
.lbm{font-size:1.6rem;width:40px;text-align:center;filter:drop-shadow(0 2px 6px rgba(0,0,0,.4));}
.lbn{flex:1;font-weight:800;color:#E2E8F0!important;font-size:.93rem;font-family:'Space Grotesk',sans-serif;}
.lbp{width:150px;height:5px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden;}
.lbpf{height:100%;border-radius:3px;background:linear-gradient(90deg,#0891B2,#06B6D4,#8B5CF6,#06B6D4,#0891B2);background-size:400% auto;animation:flowGlow 3s linear infinite;}
.lbm2{font-size:.7rem;color:#64748B!important;width:120px;text-align:right;font-weight:600;font-family:'JetBrains Mono',monospace;}
.lba{font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:900;background:linear-gradient(135deg,#00D4FF,#8B5CF6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;width:72px;text-align:right;}

/* ── SECTION HEADER ── */
.sh{
  display:flex;align-items:center;gap:10px;
  background:linear-gradient(90deg,rgba(0,212,255,.08),rgba(0,212,255,.02),transparent);
  border-left:3px solid #06B6D4;border-radius:0 8px 8px 0;
  padding:10px 16px;margin:28px 0 14px;
  font-weight:700;font-size:.82rem;color:#00D4FF!important;
  letter-spacing:.1em;text-transform:uppercase;
  font-family:'JetBrains Mono',monospace;
  box-shadow:-3px 0 20px rgba(0,212,255,.1);
}

/* ── EXPANDER ── */
div[data-testid="stExpander"]{background:rgba(0,212,255,.02)!important;border:1px solid rgba(0,212,255,.09)!important;border-radius:12px!important;overflow:hidden!important;box-shadow:0 2px 12px rgba(0,0,0,.25)!important;transition:border-color .2s,box-shadow .2s!important;}
div[data-testid="stExpander"]:hover{border-color:rgba(0,212,255,.24)!important;box-shadow:0 0 20px rgba(0,212,255,.08)!important;}

/* ── SPINNER ── */
div[data-testid="stSpinner"] p{background:linear-gradient(135deg,#00D4FF,#8B5CF6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700;font-family:'JetBrains Mono',monospace;}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:rgba(0,0,0,.2);border-radius:3px;}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,#0891B2,#8B5CF6);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#06B6D4;}

/* ── IMAGES ── */
img{border-radius:12px;box-shadow:0 6px 28px rgba(0,0,0,.4),0 0 1px rgba(0,212,255,.1)!important;transition:transform .3s,box-shadow .3s!important;border:1px solid rgba(0,212,255,.07)!important;}
img:hover{transform:scale(1.02)!important;box-shadow:0 12px 48px rgba(0,0,0,.4),0 0 24px rgba(0,212,255,.12)!important;}

/* ── DIVIDER ── */
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,rgba(0,212,255,.3),rgba(139,92,246,.3),rgba(0,212,255,.3),transparent)!important;margin:28px 0!important;}

/* ── PLOTLY ── */
div[data-testid="stPlotlyChart"]{border-radius:14px!important;overflow:hidden!important;border:1px solid rgba(0,212,255,.07)!important;box-shadow:0 4px 24px rgba(0,0,0,.3),0 0 20px rgba(0,212,255,.04)!important;}

/* ── PAGE FADE ── */
div[data-testid="stVerticalBlock"]{animation:pageFade .28s ease-out both;}
@keyframes pageFade{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

/* ── RIPPLE ── */
.ripple{position:absolute;border-radius:50%;background:rgba(0,212,255,.28);transform:scale(0);animation:rippleAnim .52s linear;pointer-events:none;z-index:9999;}
@keyframes rippleAnim{to{transform:scale(5);opacity:0;}}

/* ══════════════════════════════════════════
   MOBILE RESPONSIVE — telefon uchun
══════════════════════════════════════════ */
@media (max-width:768px){
  /* Hero */
  .hero{padding:28px 20px 24px!important;margin-bottom:20px!important;}
  .hero>div[style*="right:56px"]{display:none!important;}
  .htitle{font-size:1.9rem!important;line-height:1.1!important;}
  .hsub{font-size:.9rem!important;}
  .eyebrow{font-size:.6rem!important;padding:5px 14px!important;}
  .chips{gap:6px!important;margin-top:14px!important;}
  .chip{font-size:.68rem!important;padding:4px 12px!important;}

  /* Stat cards — 2 ustun */
  .stat{padding:18px 12px!important;border-radius:12px!important;}
  .si{font-size:1.8rem!important;margin-bottom:8px!important;}
  .sv{font-size:1.4rem!important;}
  .sl{font-size:.58rem!important;}

  /* Metric */
  div[data-testid="metric-container"]{padding:14px!important;}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"]{font-size:1.3rem!important;}
  div[data-testid="metric-container"] label{font-size:.6rem!important;}

  /* Sidebar — mobilda pastda tugma */
  section[data-testid="stSidebar"]{
    max-width:100%!important;
    width:100%!important;
  }
  section[data-testid="stSidebar"] .stRadio label{
    padding:10px 12px!important;
    font-size:.8rem!important;
  }

  /* Buttons */
  div.stButton>button[kind="primary"]{
    padding:14px 20px!important;
    font-size:.82rem!important;
  }

  /* Leaderboard */
  .lb{padding:10px 12px!important;gap:8px!important;}
  .lbp{width:80px!important;}
  .lbm2{width:80px!important;font-size:.62rem!important;}
  .lba{font-size:.9rem!important;width:52px!important;}
  .lbn{font-size:.8rem!important;}

  /* Probability bars */
  .pbf{font-size:.7rem!important;}

  /* Result badge */
  .rbadge{font-size:1.1rem!important;padding:20px 16px!important;}

  /* Section header */
  .sh{font-size:.72rem!important;padding:8px 12px!important;}

  /* Charts full width */
  div[data-testid="stPlotlyChart"]{border-radius:10px!important;}

  /* Hero floating icons — yashirish */
  .hero div[style*="right:56px"]{display:none!important;}

  /* Tabs scroll */
  div[data-baseweb="tab-list"]{overflow-x:auto!important;flex-wrap:nowrap!important;}
  div[data-baseweb="tab"]{white-space:nowrap!important;font-size:.78rem!important;padding:8px 12px!important;}

  /* Number input */
  div[data-testid="stNumberInput"] input{font-size:.9rem!important;}

  /* DataFrame */
  div[data-testid="stDataFrame"]{font-size:.78rem!important;}
}

@media (max-width:480px){
  .htitle{font-size:1.55rem!important;}
  .hero{padding:22px 14px 20px!important;}
  .sv{font-size:1.2rem!important;}
  .si{font-size:1.5rem!important;}
  .rbadge{font-size:.95rem!important;padding:16px 12px!important;letter-spacing:.02em!important;}
  .lbp{display:none!important;}
  div[data-baseweb="tab"]{font-size:.72rem!important;padding:7px 9px!important;}
}
</style>

<script>
(function(){
  function addRipple(e){
    const btn = e.target.closest('button');
    if(!btn) return;
    const r = document.createElement('span');
    r.classList.add('ripple');
    const rect = btn.getBoundingClientRect();
    const sz = Math.max(rect.width, rect.height);
    r.style.cssText = `width:${sz}px;height:${sz}px;left:${e.clientX-rect.left-sz/2}px;top:${e.clientY-rect.top-sz/2}px;`;
    btn.appendChild(r);
    r.addEventListener('animationend',()=>r.remove());
  }
  document.addEventListener('click', addRipple);
})();
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════ CONSTANTS
MODELS  = ["XGBoost","LightGBM","Gradient Boosting","Random Forest",
           "Logistic Regression","SVM","KNN","Neural Network","Ensemble"]
CV_PATH = os.path.join(BASE, "reports", "cv_results.json")
CLASSES = ["Dropout","Enrolled","Graduate"]
CCOL    = {"Dropout":"#F87171","Enrolled":"#FBBF24","Graduate":"#34D399"}
MEDALS  = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣"]

# ══════════════════════════════════════ HELPERS
@st.cache_data
def load_df():
    if not os.path.exists(DATA_PATH):
        with st.spinner("📥 Dataset yuklanmoqda (UCI Repository)..."):
            try:
                from ucimlrepo import fetch_ucirepo
                ds = fetch_ucirepo(id=697)
                df = pd.concat([ds.data.features, ds.data.targets], axis=1)
                os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
                df.to_csv(DATA_PATH, index=False)
            except Exception as e:
                st.error(f"❌ Dataset yuklab bo'lmadi: {e}")
                st.stop()
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_mdl(name):
    p = os.path.join(MODELS_DIR, f"{name.replace(' ','_').lower()}.pkl")
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data
def load_res():
    if not os.path.exists(REPORT_PATH): return {}
    with open(REPORT_PATH) as f: return json.load(f)

def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def dark_fig():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,8,16,.8)",
        font=dict(color="#00D4FF", family="JetBrains Mono"),
        margin=dict(l=8, r=8, t=42, b=8),
    )

def dark_axes(fig):
    ax = dict(gridcolor="rgba(0,212,255,.07)", zerolinecolor="rgba(0,212,255,.18)", color="#0891B2")
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    return fig

def apply_dark(fig, **kw):
    fig.update_layout(**dark_fig(), **kw)
    dark_axes(fig)
    return fig

@st.cache_data
def get_sc_feat():
    from src.preprocessor import _engineer_features
    from sklearn.model_selection import train_test_split
    df_full = load_df()
    y = df_full["Target"]
    eng = _engineer_features(df_full.drop(columns=["Target"], errors="ignore"))
    # Fit scaler on train split only — matches original training procedure (random_state=42, 80/20)
    X_train, _ = train_test_split(eng, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(X_train)
    return sc, list(eng.columns)

def run_predict(model, df_raw):
    from src.preprocessor import _engineer_features
    sc, feat = get_sc_feat()
    eng = _engineer_features(df_raw.copy())
    for c in feat:
        if c not in eng.columns:
            eng[c] = 0
    X = sc.transform(eng[feat])
    return model.predict(X), model.predict_proba(X)

# ══════════════════════════════════════ SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="padding:28px 16px 22px;text-align:center;
      border-bottom:1px solid rgba(0,212,255,.1);margin-bottom:12px;
      position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:3px;
        background:linear-gradient(90deg,transparent,#06B6D4,#8B5CF6,#06B6D4,transparent);
        background-size:400% auto;animation:topBar 3s linear infinite;"></div>
      <div style="font-size:3rem;animation:logoBob 3s ease-in-out infinite;display:inline-block;
        filter:drop-shadow(0 0 14px rgba(0,212,255,.5));">🎓</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.25rem;font-weight:900;
        background:linear-gradient(135deg,#00D4FF,#8B5CF6,#00D4FF);background-size:300% auto;
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        margin-top:10px;letter-spacing:.02em;text-transform:uppercase;
        animation:logoGrad 5s ease-in-out infinite alternate;">EduPredict AI</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#1E293B;margin-top:5px;
        letter-spacing:.16em;text-transform:uppercase;font-weight:600;">
        Student Dropout Prediction
      </div>
      <div style="margin-top:14px;display:inline-flex;align-items:center;gap:8px;
        background:rgba(0,212,255,.07);border:1px solid rgba(0,212,255,.2);border-radius:6px;
        padding:5px 16px;">
        <span style="display:inline-block;width:7px;height:7px;background:#06B6D4;border-radius:50%;
          box-shadow:0 0 8px rgba(6,182,212,.9),0 0 16px rgba(6,182,212,.4);
          animation:dotPulse 1.8s ease-in-out infinite;"></span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#00D4FF;font-weight:700;letter-spacing:.12em;">ONLINE</span>
      </div>
    </div>
    <style>
      @keyframes topBar{0%{background-position:0% 0}100%{background-position:400% 0}}
      @keyframes logoBob{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
      @keyframes logoGrad{0%{background-position:0% 50%}100%{background-position:100% 50%}}
      @keyframes dotPulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.25;transform:scale(.5)}}
    </style>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Sahifalar",
        ["🏠  Dashboard",
         "📊  Data Explorer",
         "🤖  Model Results",
         "🔮  Predict",
         "📤  Batch Predict",
         "📉  Risk Monitor",
         "🔬  SHAP Analysis",
         "📋  Cross-Validation",
         "📄  Hisobot",
         "ℹ️  Loyiha Haqida"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="margin-top:20px;padding:16px;
      background:rgba(0,212,255,.02);
      border:1px solid rgba(0,212,255,.12);border-radius:12px;
      box-shadow:0 4px 20px rgba(0,0,0,.3),0 0 20px rgba(0,212,255,.04);
      position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
        background:linear-gradient(90deg,transparent,#06B6D4,#8B5CF6,#06B6D4,transparent);
        background-size:400% auto;animation:topBar 3s linear infinite;"></div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;font-weight:700;
        color:#00D4FF;letter-spacing:.14em;text-transform:uppercase;margin-bottom:12px;">
        // DATASET INFO</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;">
        <div style="background:rgba(0,212,255,.05);border-radius:8px;padding:9px 11px;
          border:1px solid rgba(0,212,255,.14);">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;font-weight:600;">Dataset</div>
          <div style="font-size:.82rem;font-weight:800;color:#00D4FF;margin-top:3px;font-family:'Space Grotesk',sans-serif;">UCI 697</div>
        </div>
        <div style="background:rgba(0,212,255,.05);border-radius:8px;padding:9px 11px;
          border:1px solid rgba(0,212,255,.14);">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;font-weight:600;">Talabalar</div>
          <div style="font-size:.82rem;font-weight:800;color:#E2E8F0;margin-top:3px;font-family:'Space Grotesk',sans-serif;">4,424</div>
        </div>
        <div style="background:rgba(139,92,246,.05);border-radius:8px;padding:9px 11px;
          border:1px solid rgba(139,92,246,.18);">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;font-weight:600;">Features</div>
          <div style="font-size:.82rem;font-weight:800;color:#A78BFA;margin-top:3px;font-family:'Space Grotesk',sans-serif;">36 + 7</div>
        </div>
        <div style="background:rgba(139,92,246,.05);border-radius:8px;padding:9px 11px;
          border:1px solid rgba(139,92,246,.18);">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;font-weight:600;">Models</div>
          <div style="font-size:.82rem;font-weight:800;color:#A78BFA;margin-top:3px;font-family:'Space Grotesk',sans-serif;">9 ML</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# ════════════════════════════════════════════════════
#  1 · DASHBOARD
# ════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">🏆 Diplom loyihasi &nbsp;·&nbsp; 2026</div>
      <div class="htitle">Talabalar Natijasini<br>Bashorat Qilish Tizimi</div>
      <div class="hsub">Machine Learning yordamida talabaning akademik muvaffaqiyatini,
        dropout xavfini va bitirish ehtimolini real vaqtda oldindan aniqlash.</div>
      <div class="chips">
        <span class="chip">🤖 9 ML Model</span>
        <span class="chip">🧬 SMOTE Balans</span>
        <span class="chip">🔬 SHAP · XAI</span>
        <span class="chip">📤 Batch Bashorat</span>
        <span class="chip">📉 Risk Monitor</span>
        <span class="chip">🗳️ Soft Voting Ensemble</span>
        <span class="chip">📋 Cross-Validation</span>
        <span class="chip">📄 HTML Hisobot</span>
      </div>
    </div>
    <div style="position:absolute;right:56px;top:50%;transform:translateY(-50%);
      display:flex;flex-direction:column;align-items:center;gap:16px;opacity:.9;">
      <div style="width:64px;height:64px;border-radius:20px;
        background:rgba(255,255,255,.25);
        border:2px solid rgba(255,255,255,.55);display:flex;align-items:center;justify-content:center;
        font-size:1.7rem;animation:orb1 4s ease-in-out infinite;
        box-shadow:0 8px 24px rgba(0,0,0,.2),inset 0 1px 0 rgba(255,255,255,.3);
        backdrop-filter:blur(10px);">📊</div>
      <div style="width:50px;height:50px;border-radius:15px;
        background:rgba(255,255,255,.2);
        border:2px solid rgba(255,255,255,.45);display:flex;align-items:center;justify-content:center;
        font-size:1.35rem;animation:orb2 4s ease-in-out infinite;
        box-shadow:0 6px 18px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.25);
        backdrop-filter:blur(10px);">🤖</div>
      <div style="width:58px;height:58px;border-radius:18px;
        background:rgba(255,255,255,.28);
        border:2px solid rgba(255,255,255,.6);display:flex;align-items:center;justify-content:center;
        font-size:1.5rem;animation:orb3 4s ease-in-out infinite;
        box-shadow:0 8px 20px rgba(0,0,0,.2),inset 0 1px 0 rgba(255,255,255,.3);
        backdrop-filter:blur(10px);">🎯</div>
    </div>
    <style>
    @keyframes orb1{0%,100%{transform:translateY(0)}33%{transform:translateY(-8px)}}
    @keyframes orb2{0%,100%{transform:translateY(0)}66%{transform:translateY(-6px)}}
    @keyframes orb3{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
    </style>
    </div>""", unsafe_allow_html=True)

    df = load_df()
    res = load_res()
    best_name = max(res, key=lambda k: res[k].get("accuracy", 0)) if res else "XGBoost"
    best_acc  = res.get(best_name, {}).get("accuracy", 0)
    best_auc  = res.get(best_name, {}).get("auc") or 0

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, ico, val, lbl in [
        (c1,"👥",f"{len(df):,}","Jami talabalar"),
        (c2,"📐",f"{df.shape[1]-1}","Xususiyatlar"),
        (c3,"🏆",best_name,"Eng yaxshi model"),
        (c4,"🎯",f"{best_acc:.1%}","Eng yuqori Accuracy"),
        (c5,"🔵",f"{best_auc:.3f}","Eng yuqori AUC"),
    ]:
        col.markdown(
            f"<div class='stat'><span class='si'>{ico}</span>"
            f"<div class='sv'>{val}</div><div class='sl'>{lbl}</div></div>",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        try:
            counts = df["Target"].value_counts()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(counts.index), y=list(counts.values),
                marker_color=[CCOL.get(c, "#94A3B8") for c in counts.index],
                text=list(counts.values), textposition="outside",
                textfont=dict(color="#C4B5FD", size=14),
            ))
            apply_dark(fig, title="Sinflar taqsimoti", height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="dash_bar")
        except Exception as e:
            st.error(f"Bar chart xatosi: {e}")

    with cr:
        try:
            fig2 = go.Figure(go.Pie(
                labels=list(counts.index), values=list(counts.values),
                marker=dict(colors=[CCOL[c] for c in counts.index],
                            line=dict(color="rgba(0,0,0,.4)", width=3)),
                hole=.42, textfont=dict(size=13),
                hovertemplate="<b>%{label}</b><br>%{value} talaba<br>%{percent}<extra></extra>",
            ))
            fig2.update_layout(**dark_fig(), title="Foiz taqsimoti", height=340,
                               legend=dict(font=dict(color="#A78BFA")))
            st.plotly_chart(fig2, use_container_width=True, key="dash_pie")
        except Exception as e:
            st.error(f"Pie chart xatosi: {e}")

    if res:
        try:
            df_ov = pd.DataFrame(
                [(n, m.get("accuracy",0), m.get("auc") or 0) for n,m in res.items()],
                columns=["Model","Accuracy","AUC"]
            ).sort_values("Accuracy", ascending=False)
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(name="Accuracy", x=df_ov["Model"], y=df_ov["Accuracy"],
                marker_color="#7828ff", text=df_ov["Accuracy"].apply(lambda x: f"{x:.2%}"),
                textposition="outside", textfont=dict(color="#8B5CF6")))
            fig3.add_trace(go.Bar(name="ROC-AUC", x=df_ov["Model"], y=df_ov["AUC"],
                marker_color="#0ea5e9", text=df_ov["AUC"].apply(lambda x: f"{x:.3f}"),
                textposition="outside", textfont=dict(color="#94A3B8")))
            apply_dark(fig3, title="Barcha modellar ko'rsatkichlari",
                       barmode="group", height=380, legend=dict(font=dict(color="#A78BFA")))
            st.plotly_chart(fig3, use_container_width=True, key="dash_models")
        except Exception as e:
            st.error(f"Model chart xatosi: {e}")



# ════════════════════════════════════════════════════
#  2 · DATA EXPLORER
# ════════════════════════════════════════════════════
elif "Data Explorer" in page:
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">📊 EDA</div>
      <div class="htitle">Ma'lumotlar Tahlili</div>
      <div class="hsub">Dataset tuzilmasi, taqsimot va korrelyatsiyalarni o'rganish</div>
    </div></div>""", unsafe_allow_html=True)

    df = load_df()
    t1,t2,t3,t4,t5 = st.tabs(["📋 Dataset","📊 Taqsimot","📈 Korrelyatsiya","📦 Sinf Tahlili","🖼️ Grafiklar"])

    with t1:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Qatorlar", f"{len(df):,}")
        m2.metric("Ustunlar", df.shape[1])
        m3.metric("Raqamli ustun", df.select_dtypes(include=np.number).shape[1])
        m4.metric("Bo'sh qiymat", df.isnull().sum().sum())
        n = st.slider("Ko'rsatiladigan qatorlar", 10, 300, 50, key="eda_slider")
        st.dataframe(df.head(n), use_container_width=True, height=380)

    with t2:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        col_sel = st.selectbox("Ustun tanlang:", num_cols, key="eda_col")
        try:
            fig = go.Figure()
            for cls, color in CCOL.items():
                sub = df[df["Target"] == cls][col_sel]
                if not sub.empty:
                    fig.add_trace(go.Histogram(x=sub, name=cls, marker_color=color,
                        opacity=.75, nbinsx=30))
            apply_dark(fig, barmode="overlay", title=f"{col_sel} — taqsimot", height=360)
            st.plotly_chart(fig, use_container_width=True, key="eda_hist")

            fig2 = go.Figure()
            for cls, color in CCOL.items():
                sub = df[df["Target"] == cls][col_sel]
                if not sub.empty:
                    fig2.add_trace(go.Box(y=sub, name=cls, marker_color=color,
                        line_color=color, notched=True, boxmean=True))
            apply_dark(fig2, title=f"{col_sel} — Boxplot", height=340)
            st.plotly_chart(fig2, use_container_width=True, key="eda_box")
        except Exception as e:
            st.error(f"Chart xatosi: {e}")

    with t3:
        try:
            num_df = df.select_dtypes(include=np.number).copy()
            num_df["Target_enc"] = LabelEncoder().fit_transform(df["Target"])
            corr = num_df.corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=list(corr.columns), y=list(corr.index),
                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
                colorbar=dict(tickfont=dict(color="#94A3B8")),
            ))
            fig.update_layout(**dark_fig(), title="Korrelyatsiya matritsasi", height=680)
            fig.update_xaxes(tickangle=-45, tickfont=dict(size=8, color="#94A3B8"),
                             gridcolor="rgba(139,92,246,.08)")
            fig.update_yaxes(tickfont=dict(size=8, color="#94A3B8"),
                             gridcolor="rgba(139,92,246,.08)")
            st.plotly_chart(fig, use_container_width=True, key="eda_heatmap")

            corr_t = num_df.corr()["Target_enc"].drop("Target_enc").sort_values(key=abs, ascending=True)
            fig2 = go.Figure(go.Bar(
                x=corr_t.values[-20:], y=list(corr_t.index[-20:]), orientation="h",
                marker_color=["#ff2d55" if v < 0 else "#00c853" for v in corr_t.values[-20:]],
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig2.add_vline(x=0, line_color="rgba(255,255,255,.3)")
            apply_dark(fig2, title="Target bilan korrelyatsiya (Top 20)", height=500)
            st.plotly_chart(fig2, use_container_width=True, key="eda_corr_bar")
        except Exception as e:
            st.error(f"Korrelyatsiya xatosi: {e}")

    with t4:
        feat_opts = [c for c in [
            "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
            "Curricular units 1st sem (approved)", "Age at enrollment", "Admission grade",
        ] if c in df.columns]
        feat4 = st.selectbox("Ko'rsatkich:", feat_opts, key="eda_feat4")
        try:
            fig = go.Figure()
            for cls, color in CCOL.items():
                sub = df[df["Target"] == cls][feat4]
                fig.add_trace(go.Violin(x=[cls]*len(sub), y=sub, name=cls,
                    line_color=color,
                    fillcolor="rgba({},{},{},0.15)".format(
                        int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                    ),
                    meanline_visible=True, box_visible=True))
            apply_dark(fig, title=f"{feat4} — Violin + Box", height=380)
            st.plotly_chart(fig, use_container_width=True, key="eda_violin")

            fig2 = make_subplots(rows=1, cols=3,
                subplot_titles=list(CCOL.keys()),
                horizontal_spacing=0.06)
            for i, (cls, color) in enumerate(CCOL.items()):
                sub = df[df["Target"] == cls][feat4]
                fig2.add_trace(go.Histogram(x=sub, marker_color=color,
                    opacity=.8, showlegend=False, nbinsx=25), row=1, col=i+1)
            fig2.update_layout(**dark_fig(),
                title=f"{feat4} — Sinf bo'yicha taqsimot", height=320)
            dark_axes(fig2)
            st.plotly_chart(fig2, use_container_width=True, key="eda_subplots")
        except Exception as e:
            st.error(f"Violin/boxplot xatosi: {e}")

    with t5:
        if os.path.exists(FIGURES_DIR):
            figs = sorted(f for f in os.listdir(FIGURES_DIR) if f.endswith(".png"))
            if figs:
                cols = st.columns(2)
                for i, fn in enumerate(figs):
                    cols[i % 2].image(
                        os.path.join(FIGURES_DIR, fn),
                        caption=fn.replace(".png","").replace("_"," ").title(),
                        use_container_width=True)
            else:
                st.info("📊 Grafiklar mavjud emas.")


# ════════════════════════════════════════════════════
#  3 · MODEL RESULTS
# ════════════════════════════════════════════════════
elif "Model Results" in page:
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">🤖 ML Modellar</div>
      <div class="htitle">Model Natijalari</div>
      <div class="hsub">9 ta ML model o'qitildi, baholandi va taqqoslandi</div>
    </div></div>""", unsafe_allow_html=True)

    res = load_res()
    if not res:
        st.warning("⚠️ Ma'lumotlar topilmadi.")
        st.stop()

    rows = [{"Model":n,"Accuracy":m.get("accuracy",0),"F1":m.get("f1",0),
             "Precision":m.get("precision",0),"Recall":m.get("recall",0),
             "ROC-AUC":m.get("auc") or 0} for n,m in res.items()]
    df_r = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🥇 Champion", df_r.iloc[0]["Model"])
    c2.metric("🎯 Accuracy", f"{df_r.iloc[0]['Accuracy']:.2%}")
    c3.metric("📊 F1 Score", f"{df_r.iloc[0]['F1']:.4f}")
    c4.metric("🔵 ROC-AUC",  f"{df_r.iloc[0]['ROC-AUC']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    cls_map = {0:"gold", 1:"silver", 2:"bronze"}
    for i, row in df_r.iterrows():
        medal = MEDALS[i] if i < len(MEDALS) else str(i+1)
        st.markdown(
            f"<div class='lb {cls_map.get(i,'')}'>"
            f"<div class='lbm'>{medal}</div>"
            f"<div class='lbn'>{row['Model']}</div>"
            f"<div class='lbp'><div class='lbpf' style='width:{int(row['Accuracy']*100)}%'></div></div>"
            f"<div class='lbm2'>F1:{row['F1']:.3f} · AUC:{row['ROC-AUC']:.3f}</div>"
            f"<div class='lba'>{row['Accuracy']:.2%}</div></div>",
            unsafe_allow_html=True)


    st.markdown("---")
    st.markdown("<div class='sh'>🔍 Model tafsilotlari</div>", unsafe_allow_html=True)
    sel = st.selectbox("Model tanlang:", MODELS, key="mr_sel")
    m   = res.get(sel, {})

    mc1,mc2,mc3,mc4,mc5 = st.columns(5)
    for col, lbl, key, fmt in [
        (mc1,"Accuracy","accuracy",".2%"), (mc2,"F1","f1",".4f"),
        (mc3,"Precision","precision",".4f"), (mc4,"Recall","recall",".4f"),
        (mc5,"ROC-AUC","auc",".4f"),
    ]:
        col.metric(lbl, format(m.get(key) or 0, fmt))

    safe = sel.replace(" ","_").lower()
    col_a, col_b = st.columns(2)
    any_fig_shown = False
    for col, prefix, title in [
        (col_a, "07_confusion_matrix_", "Confusion Matrix"),
        (col_b, "09_roc_curves_",       "ROC Curves"),
    ]:
        p = os.path.join(FIGURES_DIR, f"{prefix}{safe}.png")
        if os.path.exists(p):
            col.image(p, caption=title, use_container_width=True)
            any_fig_shown = True

    for prefix, title in [
        ("08_feature_importance_", "Feature Importance"),
        ("10_learning_curve_",     "Learning Curve"),
    ]:
        p = os.path.join(FIGURES_DIR, f"{prefix}{safe}.png")
        if os.path.exists(p):
            st.image(p, caption=title, use_container_width=True)
            any_fig_shown = True

    if not any_fig_shown:
        st.info("📊 Grafiklar mavjud emas.")

    # Per-class metrics figure
    pc_fig = os.path.join(FIGURES_DIR, f"11_per_class_{safe}.png")
    if os.path.exists(pc_fig):
        st.image(pc_fig, caption="Per-Class Metrics (Precision / Recall / F1)", use_container_width=True)

    # Per-class metrics table from JSON
    PC_PATH = os.path.join(BASE, "reports", "per_class_results.json")
    if os.path.exists(PC_PATH):
        with open(PC_PATH) as f_pc:
            pc_data = json.load(f_pc)
        if sel in pc_data:
            st.markdown("<div class='sh'>📋 Sinf bo'yicha batafsil metrikalar</div>", unsafe_allow_html=True)
            pc_rep = pc_data[sel]
            rows_pc = []
            for cls in CLASSES:
                if cls in pc_rep and isinstance(pc_rep[cls], dict):
                    rows_pc.append({
                        "Sinf": cls,
                        "Precision": round(pc_rep[cls].get("precision", 0), 4),
                        "Recall":    round(pc_rep[cls].get("recall", 0), 4),
                        "F1-Score":  round(pc_rep[cls].get("f1-score", 0), 4),
                        "Support":   int(pc_rep[cls].get("support", 0)),
                    })
            if rows_pc:
                df_pc = pd.DataFrame(rows_pc)
                st.dataframe(
                    df_pc.style
                    .background_gradient(subset=["F1-Score"], cmap="Greens")
                    .background_gradient(subset=["Recall"],    cmap="Blues")
                    .format({"Precision":"{:.4f}","Recall":"{:.4f}","F1-Score":"{:.4f}"}),
                    use_container_width=True, hide_index=True)

    pkl = os.path.join(MODELS_DIR, f"{safe}.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            st.download_button(
                f"⬇️  {sel} modelini .pkl yuklab olish",
                data=f.read(),
                file_name=f"{safe}.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                key=f"mr_pkl_{safe}")


# ════════════════════════════════════════════════════
#  4 · PREDICT
# ════════════════════════════════════════════════════
elif "Predict" in page and "Batch" not in page:
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">🔮 AI Bashorat</div>
      <div class="htitle">Talaba Holatini Aniqlash</div>
      <div class="hsub">Ma'lumotlarni kiriting — model darhol Dropout · Enrolled · Graduate bashorat qiladi</div>
    </div></div>""", unsafe_allow_html=True)

    sel_m = st.selectbox("🤖 Model:", MODELS, key="pred_model")
    model = load_mdl(sel_m)
    if not model:
        st.error("❌ Model fayllari topilmadi.")
        st.stop()

    # ── Demo profiles
    DEMOS = {
        "graduate": dict(
            p_age=19, p_gender=0, p_debtor=0, p_tuition=1, p_schol=1,
            s1en=6, s1ap=6, s1gr=14.0, s2en=6, s2ap=6, s2gr=14.5,
            p_adg=155.0, p_marital=2, p_nation=1, p_intl=0, p_displ=0,
            p_spcn=0, p_mq=19, p_fq=12, p_mo=10, p_fo=9,
            p_amode=1, p_aord=1, p_crs=9500, p_att=1, p_pq=1,
            p_pqg=160.0, s1cr=0, s1ev=8, s1ne=0, s2cr=0, s2ev=7, s2ne=0,
            p_unemp=10.8, p_infl=1.4, p_gdp=1.74,
        ),
        "dropout": dict(
            p_age=24, p_gender=1, p_debtor=1, p_tuition=0, p_schol=0,
            s1en=4, s1ap=0, s1gr=0.0, s2en=5, s2ap=0, s2gr=0.0,
            p_adg=105.0, p_marital=1, p_nation=1, p_intl=0, p_displ=1,
            p_spcn=0, p_mq=2, p_fq=2, p_mo=9, p_fo=9,
            p_amode=17, p_aord=5, p_crs=9238, p_att=0, p_pq=1,
            p_pqg=102.0, s1cr=0, s1ev=2, s1ne=2, s2cr=0, s2ev=1, s2ne=3,
            p_unemp=13.9, p_infl=2.8, p_gdp=-2.1,
        ),
        "enrolled": dict(
            p_age=26, p_gender=1, p_debtor=1, p_tuition=1, p_schol=0,
            s1en=6, s1ap=0, s1gr=0.0, s2en=6, s2ap=0, s2gr=0.0,
            p_adg=140.0, p_marital=1, p_nation=1, p_intl=0, p_displ=1,
            p_spcn=0, p_mq=38, p_fq=1, p_mo=134, p_fo=134,
            p_amode=44, p_aord=1, p_crs=9003, p_att=1, p_pq=39,
            p_pqg=140.0, s1cr=0, s1ev=6, s1ne=0, s2cr=0, s2ev=7, s2ne=0,
            p_unemp=8.9, p_infl=1.4, p_gdp=3.51,
        ),
    }

    # ── Demo buttons
    st.markdown("<div class='sh'>⚡ Demo namunalar — bir bosishda sinab ko'ring</div>",
                unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    if dc1.button("🎓 A'lo talaba (Graduate)", use_container_width=True, key="demo_grad"):
        for k, v in DEMOS["graduate"].items(): st.session_state[k] = v
        st.session_state["auto_predict"] = True
        st.rerun()
    if dc2.button("⚠️ Xavf ostida (Dropout)", use_container_width=True, key="demo_drop"):
        for k, v in DEMOS["dropout"].items(): st.session_state[k] = v
        st.session_state["auto_predict"] = True
        st.rerun()
    if dc3.button("📚 O'qishda davom (Enrolled)", use_container_width=True, key="demo_enrl"):
        for k, v in DEMOS["enrolled"].items(): st.session_state[k] = v
        st.session_state["auto_predict"] = True
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main form — 2 columns: left=personal+financial, right=academic
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("<div class='sh'>👤 Shaxsiy va moliyaviy</div>", unsafe_allow_html=True)
        age    = st.slider("🎂 Yosh (qabul paytida)", 17, 70, 20, key="p_age")
        gender = st.selectbox("⚧ Jins", [0, 1],
                    format_func=lambda x: "👩 Ayol" if x == 0 else "👨 Erkak",
                    key="p_gender")
        tuition = st.selectbox("💳 To'lov muddatida to'langan", [1, 0],
                    format_func=lambda x: "✅ Ha" if x == 1 else "❌ Yo'q",
                    key="p_tuition")
        schol  = st.selectbox("🎓 Stipendiyant", [0, 1],
                    format_func=lambda x: "Yo'q" if x == 0 else "✅ Ha",
                    key="p_schol")
        debtor = st.selectbox("💸 Qarzdor", [0, 1],
                    format_func=lambda x: "Yo'q" if x == 0 else "⚠️ Ha",
                    key="p_debtor")
        adg    = st.slider("📋 Qabul bahosi", 95.0, 190.0, 127.0, key="p_adg")

    with right:
        st.markdown("<div class='sh'>📖 Akademik natijalar</div>", unsafe_allow_html=True)
        st.markdown("**1-Semestr**")
        r1a, r1b, r1c = st.columns(3)
        s1en = r1a.slider("Yozilgan fan", 0, 26, 6, key="s1en")
        s1ap = r1b.slider("O'tilgan fan", 0, 26, 5, key="s1ap")
        s1gr = r1c.slider("O'rtacha baho", 0.0, 20.0, 12.0, key="s1gr")

        st.markdown("**2-Semestr**")
        r2a, r2b, r2c = st.columns(3)
        s2en = r2a.slider("Yozilgan fan", 0, 23, 6, key="s2en")
        s2ap = r2b.slider("O'tilgan fan", 0, 20, 5, key="s2ap")
        s2gr = r2c.slider("O'rtacha baho", 0.0, 20.0, 12.0, key="s2gr")

    # ── Advanced settings expander
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️ Kengaytirilgan sozlamalar — qo'shimcha maydonlar (ixtiyoriy)", expanded=False):
        st.markdown("<div class='sh'>📋 Ijtimoiy va oilaviy ma'lumotlar</div>", unsafe_allow_html=True)
        adv1, adv2, adv3 = st.columns(3)
        marital = adv1.selectbox("💍 Oilaviy holat", [1,2,3,4,5,6],
            format_func=lambda x: {1:"Yagona",2:"Turmush qurgan",3:"Keva",4:"Ajrashgan",5:"Birga yashovchi",6:"Qonuniy ajrashgan"}.get(x,str(x)),
            key="p_marital")
        displ   = adv2.selectbox("🏘️ Ko'chib kelgan", [0,1],
            format_func=lambda x: "Yo'q" if x==0 else "✅ Ha", key="p_displ")
        spcn    = adv3.selectbox("♿ Maxsus ehtiyoj", [0,1],
            format_func=lambda x: "Yo'q" if x==0 else "✅ Ha", key="p_spcn")

        adv4, adv5 = st.columns(2)
        intl    = adv4.selectbox("🌍 Xorijiy talaba", [0,1],
            format_func=lambda x: "Yo'q" if x==0 else "✅ Ha", key="p_intl")
        nation  = adv5.number_input("🗺️ Millat (kodi)", 1, 109, 1, key="p_nation")

        st.markdown("<div class='sh'>👨‍👩‍👧 Ota-ona ma'lumotlari</div>", unsafe_allow_html=True)
        oe1, oe2, oe3, oe4 = st.columns(4)
        mq = oe1.number_input("Ona ta'lim darajasi", 0, 44, 19, key="p_mq")
        fq = oe2.number_input("Ota ta'lim darajasi", 0, 44, 22, key="p_fq")
        mo = oe3.number_input("Ona kasbi (kodi)", 0, 194, 10, key="p_mo")
        fo = oe4.number_input("Ota kasbi (kodi)", 0, 194, 10, key="p_fo")

        st.markdown("<div class='sh'>🎓 Qabul va kurs ma'lumotlari</div>", unsafe_allow_html=True)
        qa1, qa2, qa3, qa4 = st.columns(4)
        amode = qa1.number_input("Ariza usuli (kodi)", 1, 57, 1, key="p_amode")
        aord  = qa2.number_input("Ariza tartibi", 0, 9, 1, key="p_aord")
        crs   = qa3.number_input("Kurs (kodi)", 0, 9999, 9500, key="p_crs")
        att   = qa4.selectbox("Dars vaqti", [1,0],
            format_func=lambda x: "🌞 Kunduzgi" if x==1 else "🌙 Kechki", key="p_att")

        qa5, qa6 = st.columns(2)
        pq  = qa5.number_input("Oldingi ta'lim turi (kodi)", 1, 43, 1, key="p_pq")
        pqg = qa6.slider("Oldingi ta'lim bahosi", 0.0, 200.0, 130.0, key="p_pqg")

        st.markdown("<div class='sh'>📖 Semestr batafsil</div>", unsafe_allow_html=True)
        sem_a, sem_b = st.columns(2)
        with sem_a:
            st.caption("1-Semestr")
            s1cr = st.number_input("Kredit (1-sem)", 0, 20, 0, key="s1cr")
            s1ev = st.number_input("Baholangan (1-sem)", 0, 45, s1en, key="s1ev")
            s1ne = st.number_input("Bahosiz (1-sem)", 0, 19, 0, key="s1ne")
        with sem_b:
            st.caption("2-Semestr")
            s2cr = st.number_input("Kredit (2-sem)", 0, 19, 0, key="s2cr")
            s2ev = st.number_input("Baholangan (2-sem)", 0, 45, s2en, key="s2ev")
            s2ne = st.number_input("Bahosiz (2-sem)", 0, 12, 0, key="s2ne")

        st.markdown("<div class='sh'>🌍 Makroiqtisodiy ko'rsatkichlar</div>", unsafe_allow_html=True)
        me1, me2, me3 = st.columns(3)
        unemp = me1.slider("📉 Ishsizlik darajasi (%)", 7.0, 17.0, 11.5, 0.1, key="p_unemp")
        infl  = me2.slider("📈 Inflyatsiya darajasi (%)", -0.8, 3.3, 1.2, 0.1, key="p_infl")
        gdp   = me3.slider("💹 YaIM o'sishi (%)", -4.1, 3.5, 0.0, 0.1, key="p_gdp")

    # ── Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    _, bc, _ = st.columns([1, 2, 1])
    go_pred   = bc.button("🔮  Bashorat Qilish", type="primary",
                          use_container_width=True, key="pred_go")
    auto_pred = st.session_state.pop("auto_predict", False)

    if go_pred or auto_pred:
        row = {
            "Marital Status": marital, "Application mode": amode,
            "Application order": aord, "Course": crs,
            "Daytime/evening attendance": att, "Previous qualification": pq,
            "Previous qualification (grade)": pqg, "Nacionality": nation,
            "Mother's qualification": mq, "Father's qualification": fq,
            "Mother's occupation": mo, "Father's occupation": fo,
            "Admission grade": adg, "Displaced": displ,
            "Educational special needs": spcn, "Debtor": debtor,
            "Tuition fees up to date": tuition, "Gender": gender,
            "Scholarship holder": schol, "Age at enrollment": age,
            "International": intl,
            "Curricular units 1st sem (credited)": s1cr,
            "Curricular units 1st sem (enrolled)": s1en,
            "Curricular units 1st sem (evaluations)": s1ev,
            "Curricular units 1st sem (approved)": s1ap,
            "Curricular units 1st sem (grade)": s1gr,
            "Curricular units 1st sem (without evaluations)": s1ne,
            "Curricular units 2nd sem (credited)": s2cr,
            "Curricular units 2nd sem (enrolled)": s2en,
            "Curricular units 2nd sem (evaluations)": s2ev,
            "Curricular units 2nd sem (approved)": s2ap,
            "Curricular units 2nd sem (grade)": s2gr,
            "Curricular units 2nd sem (without evaluations)": s2ne,
            "Unemployment rate": unemp, "Inflation rate": infl, "GDP": gdp,
        }
        try:
            preds, probs = run_predict(model, pd.DataFrame([row]))
            label = CLASSES[preds[0]]
            prob  = probs[0]
            b_cls = {"Dropout":"rb-d","Enrolled":"rb-e","Graduate":"rb-g"}[label]
            b_txt = {"Dropout":"⚠️ DROPOUT XAVFI",
                     "Enrolled":"📚 O'QISHDA DAVOM ETADI",
                     "Graduate":"🎓 MUVAFFAQIYATLI BITIRADI"}[label]

            st.markdown("---")
            _, rc, _ = st.columns([1, 2, 1])
            with rc:
                st.markdown(
                    f"<div style='text-align:center;padding:16px 0;'>"
                    f"<div style='font-size:.8rem;color:#8B5CF6;margin-bottom:12px;"
                    f"letter-spacing:.08em;text-transform:uppercase;font-weight:700;'>"
                    f"Bashorat natijasi · {sel_m}</div>"
                    f"<div class='rbadge {b_cls}'>{b_txt}</div>"
                    f"</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            cl2, cr2 = st.columns(2)
            with cl2:
                for cls, p in zip(CLASSES, prob):
                    st.markdown(
                        f"<div class='pb'>"
                        f"<div class='pbh'><span style='font-weight:700'>{cls}</span>"
                        f"<span style='font-size:1.1rem;font-weight:800;color:{CCOL[cls]}'>{p:.1%}</span></div>"
                        f"<div class='pbb'><div class='pbf' "
                        f"style='width:{int(p*100)}%;background:{CCOL[cls]};'></div></div>"
                        f"</div>", unsafe_allow_html=True)
            with cr2:
                try:
                    fig_p = go.Figure(go.Pie(
                        labels=CLASSES, values=[float(p) for p in prob],
                        marker=dict(colors=[CCOL[c] for c in CLASSES],
                                    line=dict(color="rgba(0,0,0,.4)", width=2)),
                        hole=.5, textfont=dict(size=12),
                        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
                    ))
                    fig_p.update_layout(**dark_fig(), height=240,
                                        legend=dict(font=dict(color="#A78BFA")),
                                        margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig_p, use_container_width=True, key="pred_pie")
                except Exception:
                    pass

            tip = {
                "Dropout": "⚠️ **Xavf darajasi: YUQORI** — Talabaga shoshilinch akademik va moliyaviy yordam kerak.",
                "Enrolled": "📚 **Holat: KUZATUVDA** — Talaba o'qishda. Motivatsiya va doimiy monitoring tavsiya etiladi.",
                "Graduate": "🎓 **Holat: A'LO** — Talaba muvaffaqiyatli bitirish yo'lida. Mavjud sharoitlarni saqlang.",
            }
            st.info(tip[label])

            # ── Barcha modellar taqqoslash
            st.markdown("---")
            st.markdown("<div class='sh'>🔬 Barcha modellar taqqoslash</div>", unsafe_allow_html=True)
            df_input = pd.DataFrame([row])
            all_results = []
            for mname in MODELS:
                mdl = load_mdl(mname)
                if mdl is None:
                    continue
                try:
                    mp, mprob = run_predict(mdl, df_input)
                    mlabel = CLASSES[mp[0]]
                    all_results.append({
                        "model": mname,
                        "label": mlabel,
                        "dropout": float(mprob[0][0]),
                        "enrolled": float(mprob[0][1]),
                        "graduate": float(mprob[0][2]),
                    })
                except Exception:
                    pass

            if all_results:
                fig_cmp = go.Figure()
                for cls, color in CCOL.items():
                    key_c = cls.lower()
                    fig_cmp.add_trace(go.Bar(
                        name=cls,
                        x=[r["model"] for r in all_results],
                        y=[r[key_c] for r in all_results],
                        marker_color=color,
                        text=[f"{r[key_c]:.1%}" for r in all_results],
                        textposition="outside",
                        textfont=dict(color="#C4B5FD", size=9),
                    ))
                apply_dark(fig_cmp,
                    title="Barcha modellar — Dropout · Enrolled · Graduate ehtimoli",
                    barmode="group", height=360,
                    legend=dict(font=dict(color="#A78BFA")),
                    yaxis_tickformat=".0%", yaxis_range=[0, 1.15])
                st.plotly_chart(fig_cmp, use_container_width=True, key="pred_all_models")

                # Ovoz berish
                votes = {}
                for r in all_results:
                    votes[r["label"]] = votes.get(r["label"], 0) + 1
                winner = max(votes, key=votes.get)
                vote_html = " &nbsp;·&nbsp; ".join(
                    [f"<b style='color:{CCOL[c]}'>{c}: {v} ovoz</b>" for c, v in sorted(votes.items(), key=lambda x: -x[1])]
                )
                st.markdown(
                    f"<div style='text-align:center;padding:14px;background:rgba(0,212,255,.04);"
                    f"border:1px solid rgba(0,212,255,.16);border-radius:12px;margin-top:8px;'>"
                    f"<div style='font-size:.72rem;color:#64748B;letter-spacing:.1em;text-transform:uppercase;"
                    f"font-family:JetBrains Mono,monospace;margin-bottom:6px;'>🗳️ UMUMIY OVOZ NATIJASI</div>"
                    f"<div style='font-size:.95rem;'>{vote_html}</div>"
                    f"</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Bashorat xatosi: {e}")
            st.code(traceback.format_exc())


# ════════════════════════════════════════════════════
# ════════════════════════════════════════════════════
#  5 · BATCH PREDICT
# ════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">📤 Ommaviy Bashorat</div>
      <div class="htitle">Batch Prediction</div>
      <div class="hsub">CSV faylni yuklang — barcha talabalar uchun bir vaqtda bashorat</div>
    </div></div>""", unsafe_allow_html=True)

    sel_m = st.selectbox("🤖 Model:", MODELS, key="batch_model")
    model = load_mdl(sel_m)
    if not model:
        st.error("❌ Model topilmadi.")
        st.stop()

    df_full = load_df()
    tmpl = df_full.drop(columns=["Target"]).head(5)
    st.download_button("⬇️  Namuna CSV (shablon) yuklab olish",
        data=to_csv(tmpl), file_name="template.csv", mime="text/csv",
        use_container_width=True, key="batch_tmpl")

    uploaded = st.file_uploader("📂  CSV fayl yuklang:", type=["csv"], key="batch_upload")

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ Yuklandi: **{df_up.shape[0]}** qator · **{df_up.shape[1]}** ustun")

            # ── CSV ustunlarini tekshirish
            required_cols = set(load_df().drop(columns=["Target"]).columns)
            uploaded_cols = set(df_up.columns)
            missing_cols  = required_cols - uploaded_cols
            extra_cols    = uploaded_cols - required_cols
            if missing_cols:
                st.error(f"❌ CSV da quyidagi ustunlar yetishmayapti ({len(missing_cols)} ta):\n`{', '.join(sorted(missing_cols))}`")
                st.info("💡 To'g'ri formatda shablon yuklab olish uchun yuqoridagi tugmani bosing.")
                st.stop()
            if extra_cols:
                st.warning(f"⚠️ Qo'shimcha ustunlar topildi (e'tiborga olinmaydi): `{', '.join(sorted(extra_cols))}`")
                df_up = df_up[list(required_cols)]

            # Null qiymatlarni tekshirish
            null_count = df_up.isnull().sum().sum()
            if null_count > 0:
                st.warning(f"⚠️ {null_count} ta bo'sh qiymat topildi — o'rtacha bilan to'ldiriladi.")
                df_up = df_up.fillna(df_up.median(numeric_only=True))

            st.dataframe(df_up.head(8), use_container_width=True)

            if st.button("🚀  Barchasi uchun bashorat", type="primary",
                         use_container_width=True, key="batch_run"):
                with st.spinner("⏳ Bashorat qilinmoqda..."):
                    try:
                        preds, probs = run_predict(model, df_up)
                        df_res = df_up.copy()
                        df_res["Bashorat"]       = [CLASSES[p] for p in preds]
                        df_res["Dropout_%"]      = [round(float(p[0])*100,1) for p in probs]
                        df_res["Enrolled_%"]     = [round(float(p[1])*100,1) for p in probs]
                        df_res["Graduate_%"]     = [round(float(p[2])*100,1) for p in probs]

                        cnt = pd.Series([CLASSES[p] for p in preds]).value_counts()
                        mc1,mc2,mc3 = st.columns(3)
                        mc1.metric("⚠️ Dropout xavfi", int(cnt.get("Dropout",0)))
                        mc2.metric("📚 O'qishda",       int(cnt.get("Enrolled",0)))
                        mc3.metric("🎓 Bitiradi",        int(cnt.get("Graduate",0)))

                        fig = go.Figure(go.Pie(
                            labels=list(cnt.index), values=list(cnt.values),
                            marker=dict(colors=[CCOL[c] for c in cnt.index],
                                        line=dict(color="rgba(0,0,0,.3)",width=2)),
                            hole=.4, textfont=dict(size=13)))
                        fig.update_layout(**dark_fig(), title="Bashorat taqsimoti", height=360,
                                          legend=dict(font=dict(color="#A78BFA")))
                        st.plotly_chart(fig, use_container_width=True, key="batch_pie")

                        st.dataframe(
                            df_res[["Bashorat","Dropout_%","Enrolled_%","Graduate_%"]
                                   + list(df_up.columns[:4])].head(60),
                            use_container_width=True)

                        st.download_button(
                            "⬇️  Barcha natijalarni CSV yuklab olish",
                            data=to_csv(df_res),
                            file_name="batch_natijalar.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="batch_dl_csv")

                    except Exception as e:
                        st.error(f"Bashorat xatosi: {e}")
                        st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"Fayl o'qish xatosi: {e}")

    st.markdown("---")
    st.markdown("<div class='sh'>📊 Mavjud dataset ustida test</div>", unsafe_allow_html=True)
    if st.button("🔄  To'liq dataset ustida bashorat qilish", type="secondary",
                 use_container_width=True, key="batch_full"):
        with st.spinner("⏳ Bashorat qilinmoqda..."):
            try:
                p2, pr2 = run_predict(model, df_full.drop(columns=["Target"]))
                df_t = df_full.copy()
                df_t["Bashorat"] = [CLASSES[p] for p in p2]
                acc = (df_t["Target"] == df_t["Bashorat"]).mean()
                st.success(f"✅ Umumiy aniqlik: **{acc:.2%}** ({int(acc*len(df_t))}/{len(df_t)})")
                st.dataframe(df_t[["Target","Bashorat"]].head(50), use_container_width=True)
                st.download_button(
                    "⬇️  To'liq natijalar CSV",
                    data=to_csv(df_t[["Target","Bashorat"]]),
                    file_name="full_dataset_bashorat.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="batch_full_dl")
            except Exception as e:
                st.error(f"Xato: {e}")
                st.code(traceback.format_exc())


# ════════════════════════════════════════════════════
#  6 · RISK MONITOR
# ════════════════════════════════════════════════════
elif "Risk" in page:
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">📉 Xavf Monitori</div>
      <div class="htitle">Risk Dashboard</div>
      <div class="hsub">Dropout xavfi bo'lgan talabalarni aniqlash va filtrlash</div>
    </div></div>""", unsafe_allow_html=True)

    sel_m = st.selectbox("🤖 Model:", MODELS, key="risk_model")
    model = load_mdl(sel_m)
    if not model:
        st.error("❌ Model topilmadi.")
        st.stop()

    df_full = load_df()
    with st.spinner("⏳ Barcha talabalar uchun bashorat qilinmoqda..."):
        try:
            preds, probs = run_predict(model, df_full.drop(columns=["Target"]))
        except Exception as e:
            st.error(f"Bashorat xatosi: {e}")
            st.stop()

    df_r = df_full.copy()
    df_r["Bashorat"]  = [CLASSES[p] for p in preds]
    df_r["Dropout_p"] = [float(p[0]) for p in probs]
    df_r["Togri"]     = df_r["Target"] == df_r["Bashorat"]

    st.markdown("<div class='sh'>🎛️ Filtrlar</div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    thresh     = f1.slider("Dropout chegara ehtimoli", 0.0, 1.0, 0.5, 0.05, key="risk_thresh")
    g_sel      = f2.multiselect("Jins", [0,1], default=[0,1], key="risk_gender",
                                format_func=lambda x:"Ayol" if x==0 else "Erkak")
    age_rng    = f3.slider("Yosh diapazoni", 17, 70, (17,70), key="risk_age")

    df_f = df_r[df_r["Gender"].isin(g_sel) & df_r["Age at enrollment"].between(*age_rng)]
    high = df_f[df_f["Dropout_p"] >= thresh].sort_values("Dropout_p", ascending=False)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("🔍 Filtrlangan", len(df_f))
    m2.metric("⚠️ Yuqori xavf", len(high))
    m3.metric("📊 Xavf ulushi", f"{len(high)/max(len(df_f),1)*100:.1f}%")
    m4.metric("✅ Aniqlik", f"{df_f['Togri'].mean():.2%}")

    try:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure(go.Histogram(x=df_f["Dropout_p"], nbinsx=30,
                marker=dict(color="#ff2d55", opacity=.8)))
            fig.add_vline(x=thresh, line_color="#ffd200", line_width=2.5,
                          annotation_text=f"Chegara: {thresh}",
                          annotation_font_color="#ffd200")
            apply_dark(fig, title="Dropout ehtimoli taqsimoti", height=320)
            st.plotly_chart(fig, use_container_width=True, key="risk_hist")

        with col_b:
            cnt2 = df_f["Bashorat"].value_counts()
            fig2 = go.Figure(go.Pie(
                labels=list(cnt2.index), values=list(cnt2.values),
                marker=dict(colors=[CCOL[c] for c in cnt2.index],
                            line=dict(color="rgba(0,0,0,.3)",width=2)),
                hole=.4, textfont=dict(size=13)))
            fig2.update_layout(**dark_fig(), title="Bashorat taqsimoti", height=320,
                               legend=dict(font=dict(color="#A78BFA")))
            st.plotly_chart(fig2, use_container_width=True, key="risk_pie")

        df_f2 = df_f.copy()
        df_f2["Yosh_guruh"] = pd.cut(df_f2["Age at enrollment"],
            bins=[16,20,25,30,40,71], labels=["17-20","21-25","26-30","31-40","41+"])
        age_risk = (df_f2.groupby("Yosh_guruh", observed=True)
                    .agg(Dropout_ort=("Dropout_p","mean"))
                    .reset_index())
        fig3 = go.Figure(go.Bar(
            x=age_risk["Yosh_guruh"].astype(str), y=age_risk["Dropout_ort"],
            marker=dict(color=age_risk["Dropout_ort"],
                        colorscale="RdYlGn_r", showscale=True),
            text=age_risk["Dropout_ort"].apply(lambda x: f"{x:.2f}"),
            textposition="outside", textfont=dict(color="#fff"),
        ))
        apply_dark(fig3, title="Yosh guruhlari bo'yicha Dropout ehtimoli",
                   height=320, yaxis_range=[0,1])
        st.plotly_chart(fig3, use_container_width=True, key="risk_age")

        fig4 = px.scatter(
            df_f.sample(min(400, len(df_f)), random_state=42),
            x="Age at enrollment", y="Dropout_p", color="Target",
            color_discrete_map=CCOL, opacity=.65,
            title="Yosh va Dropout ehtimoli")
        fig4.update_layout(**dark_fig(), height=340, legend=dict(font=dict(color="#A78BFA")))
        dark_axes(fig4)
        st.plotly_chart(fig4, use_container_width=True, key="risk_scatter")
    except Exception as e:
        st.error(f"Chart xatosi: {e}")

    st.markdown(f"<div class='sh'>⚠️ Yuqori xavfdagi talabalar — {len(high)} ta</div>",
                unsafe_allow_html=True)
    show = [c for c in ["Target","Bashorat","Dropout_p","Gender","Age at enrollment",
            "Scholarship holder","Debtor","Tuition fees up to date",
            "Curricular units 1st sem (approved)","Curricular units 2nd sem (approved)"]
            if c in high.columns]
    st.dataframe(high[show].head(100), use_container_width=True, height=320)



# ════════════════════════════════════════════════════
#  7 · SHAP ANALYSIS
# ════════════════════════════════════════════════════
elif "SHAP" in page:
    import pickle as _pkl
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">🔬 XAI — SHAP</div>
      <div class="htitle">SHAP Tahlili</div>
      <div class="hsub">Har bir xususiyatning bashoratga ta'sirini tushuntirish — Explainable AI</div>
    </div></div>""", unsafe_allow_html=True)

    SHAP_CACHE = os.path.join(BASE, "reports", "shap_cache.pkl")

    @st.cache_data
    def load_shap_cache():
        if not os.path.exists(SHAP_CACHE): return {}
        with open(SHAP_CACHE, "rb") as f: return _pkl.load(f)

    cache = load_shap_cache()
    available = [m for m in ["XGBoost","LightGBM","Random Forest","Gradient Boosting"] if m in cache]

    if not available:
        st.warning("⏳ SHAP cache topilmadi. `python precompute_shap.py` ni ishga tushiring.")
        st.stop()

    sel_m = st.selectbox("🤖 Model:", available, key="shap_model")
    data  = cache[sel_m]
    sv    = data["shap_values"]
    feat  = data["feature_names"]
    X_s   = data["X_sample"]
    n_samples = X_s.shape[0]

    # ── 1. Global importance
    st.markdown("<div class='sh'>🌍 Global xususiyat ta'siri (Top 20)</div>", unsafe_allow_html=True)
    if isinstance(sv, list):
        mean_abs = np.mean(np.abs(np.array(sv)), axis=(0, 1))
    else:
        mean_abs = np.mean(np.abs(sv), axis=0)

    top_i = np.argsort(mean_abs)[-20:][::-1]
    top_vals  = mean_abs[top_i]
    top_feats = [feat[i] for i in top_i]

    # Color gradient: top features red, lower ones lighter
    norm = top_vals / top_vals.max()
    bar_colors = [f"rgba({int(220-80*n)},{int(38+20*n)},{int(38+20*n)},0.85)" for n in norm]

    fig = go.Figure(go.Bar(
        x=top_vals, y=top_feats, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.4f}" for v in top_vals],
        textposition="outside", textfont=dict(color="#C4B5FD", size=10),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    apply_dark(fig, title=f"Top 20 SHAP Global Ta'sir — {sel_m} ({n_samples} namuna)",
               height=560, yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True, key="shap_global")

    # ── 2. Per-class breakdown
    if isinstance(sv, list) and len(sv) == 3:
        st.markdown("<div class='sh'>🎯 Sinf bo'yicha xususiyat ta'siri (Top 10)</div>",
                    unsafe_allow_html=True)
        cls_cols = st.columns(3)
        for ci, (cls_name, col) in enumerate(zip(CLASSES, cls_cols)):
            sv_c = np.mean(np.abs(sv[ci]), axis=0)
            ti   = np.argsort(sv_c)[-10:][::-1]
            fig2 = go.Figure(go.Bar(
                x=sv_c[ti], y=[feat[i] for i in ti], orientation="h",
                marker_color=CCOL.get(cls_name, "#8B5CF6"),
                text=[f"{sv_c[i]:.3f}" for i in ti],
                textposition="outside", textfont=dict(color="#C4B5FD", size=9),
            ))
            apply_dark(fig2, title=f"► {cls_name}", height=360,
                       yaxis_autorange="reversed")
            col.plotly_chart(fig2, use_container_width=True, key=f"shap_cls_{ci}")

    # ── 3. Feature correlation waterfall (top 8)
    st.markdown("---")
    st.markdown("<div class='sh'>📊 Xususiyat muhimlik jadvali</div>", unsafe_allow_html=True)
    top8_feats = top_feats[:8]
    top8_vals  = top_vals[:8]
    df_imp = pd.DataFrame({"Xususiyat": top8_feats, "O'rtacha |SHAP|": top8_vals})
    df_imp["Ta'sir darajasi"] = pd.cut(top8_vals,
        bins=[0, 0.05, 0.15, 0.30, 1.0],
        labels=["Past", "O'rta", "Yuqori", "Juda yuqori"])
    st.dataframe(df_imp.style.background_gradient(subset=["O'rtacha |SHAP|"], cmap="Reds"),
                 use_container_width=True, hide_index=True)

    # ── 4. Individual student SHAP
    st.markdown("---")
    st.markdown("<div class='sh'>🔍 Alohida talaba SHAP tahlili</div>", unsafe_allow_html=True)

    model_obj = load_mdl(sel_m)
    if model_obj:
        idx_s = st.slider("Talaba indeksi (0 – namuna ichidan)", 0, n_samples - 1, 0,
                          key="shap_idx")
        X_row = pd.DataFrame([X_s[idx_s]], columns=feat)
        pred_l = CLASSES[model_obj.predict(X_row)[0]]
        pred_p = model_obj.predict_proba(X_row)[0]

        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Bashorat", pred_l)
        ic2.metric("Ishonch", f"{max(pred_p):.1%}")
        ic3.metric("Model", sel_m)

        pi   = CLASSES.index(pred_l) if isinstance(sv, list) and pred_l in CLASSES else 0
        sv_i = sv[pi][idx_s] if isinstance(sv, list) else sv[idx_s]

        contrib = (pd.DataFrame({"Feature": feat, "SHAP": sv_i})
                   .sort_values("SHAP", key=abs, ascending=False).head(15))
        fig3 = go.Figure(go.Bar(
            x=contrib["SHAP"], y=contrib["Feature"], orientation="h",
            marker_color=["#8B5CF6" if v < 0 else "#16A34A" for v in contrib["SHAP"]],
            text=contrib["SHAP"].apply(lambda x: f"{x:+.4f}"),
            textposition="outside", textfont=dict(color="#C4B5FD", size=10),
        ))
        fig3.add_vline(x=0, line_color="rgba(139,92,246,.4)", line_width=2)
        apply_dark(fig3, title=f"Talaba #{idx_s} — '{pred_l}' sinfi uchun SHAP",
                   height=480, yaxis_autorange="reversed")
        st.plotly_chart(fig3, use_container_width=True, key="shap_ind")

    st.success(f"✅ SHAP tahlili tayyor — {sel_m} · {n_samples} namuna asosida")


# ════════════════════════════════════════════════════
#  11 · CROSS-VALIDATION
# ════════════════════════════════════════════════════
elif "Cross-Validation" in page:
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">📋 Cross-Validation</div>
      <div class="htitle">CV Natijalari</div>
      <div class="hsub">5-fold stratified CV — modellarning haqiqiy generalizatsiya qobiliyatini o'lchash</div>
    </div></div>""", unsafe_allow_html=True)

    @st.cache_data
    def load_cv():
        if os.path.exists(CV_PATH):
            with open(CV_PATH) as f: return json.load(f)
        return {}

    cv = load_cv()
    if not cv:
        st.warning("⚠️ CV natijalari topilmadi.")
        st.stop()

    # Summary metrics
    valid_cv = {k:v for k,v in cv.items() if v.get("mean") and not pd.isna(v["mean"])}
    df_cv = pd.DataFrame([
        {"Model": k, "CV Mean": v["mean"], "Std": v["std"],
         "Min": v["min"], "Max": v["max"]}
        for k, v in valid_cv.items()
    ]).sort_values("CV Mean", ascending=False).reset_index(drop=True)

    best_cv = df_cv.iloc[0]
    c1,c2,c3 = st.columns(3)
    c1.metric("🥇 Eng yaxshi CV", best_cv["Model"])
    c2.metric("📊 CV Mean", f"{best_cv['CV Mean']:.4f}")
    c3.metric("📉 Std (barqarorlik)", f"±{best_cv['Std']:.4f}")

    # Bar chart with error bars
    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_cv["Model"], y=df_cv["CV Mean"],
            error_y=dict(type="data", array=df_cv["Std"].tolist(), visible=True,
                         color="rgba(139,92,246,.5)", thickness=2, width=8),
            marker=dict(color=df_cv["CV Mean"], colorscale="RdYlGn", showscale=True,
                        colorbar=dict(tickfont=dict(color="#94A3B8"))),
            text=df_cv["CV Mean"].apply(lambda x: f"{x:.4f}"),
            textposition="outside", textfont=dict(color="#C4B5FD"),
        ))
        apply_dark(fig, title="5-Fold CV Accuracy (± std)", height=420, showlegend=False,
                   yaxis_range=[0, df_cv["CV Mean"].max()*1.12])
        st.plotly_chart(fig, use_container_width=True, key="cv_bar")
    except Exception as e:
        st.error(f"{e}")

    # Box plot of fold scores
    try:
        fig2 = go.Figure()
        clrs = ["#7828ff","#0ea5e9","#00c853","#ff9f00","#ff2d55","#e0aaff","#7af5ff","#ffd060","#ff8fab"]
        for i, (name, v) in enumerate(valid_cv.items()):
            scores = v.get("scores", [])
            if scores:
                fig2.add_trace(go.Box(
                    y=scores, name=name,
                    marker_color=clrs[i % len(clrs)],
                    line_color=clrs[i % len(clrs)],
                    boxmean=True,
                ))
        apply_dark(fig2, title="CV fold natijalari taqsimoti (5 fold)", height=400)
        st.plotly_chart(fig2, use_container_width=True, key="cv_box")
    except Exception as e:
        st.error(f"{e}")

    # Table
    st.markdown("<div class='sh'>📋 To'liq CV jadvali</div>", unsafe_allow_html=True)
    st.dataframe(
        df_cv.style.format({c:"{:.4f}" for c in ["CV Mean","Std","Min","Max"]})
        .background_gradient(subset=["CV Mean"], cmap="Greens")
        .bar(subset=["Std"], color="#ff2d55", vmin=0, vmax=0.05),
        use_container_width=True)


    # What is CV?
    with st.expander("ℹ️ Cross-Validation nima?"):
        st.markdown("""
        **5-Fold Stratified Cross-Validation** ishlash tartibi:

        1. Train set 5 ta teng qismga bo'linadi (stratified — sinf nisbati saqlanadi)
        2. Har iteratsiyada 4 ta qism train, 1 ta qism validation sifatida ishlatiladi
        3. 5 ta accuracy natijasi olinadi va o'rtachasi hisoblanadi
        4. Bu test set'ga o'lchanadigan oddiy accuracydan ishonchliroq ko'rsatkich beradi

        **Std (standart og'ish)** kichik bo'lsa — model barqaror, katta bo'lsa — beqaror.
        """)


# ════════════════════════════════════════════════════
#  13 · HISOBOT (HTML REPORT)
# ════════════════════════════════════════════════════
elif "Hisobot" in page:
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">📄 Avtomatik Hisobot</div>
      <div class="htitle">Hisobot Generatsiya</div>
      <div class="hsub">Loyiha natijalarini HTML yoki JSON formatda yuklab oling</div>
    </div></div>""", unsafe_allow_html=True)

    res  = load_res()
    df   = load_df()

    @st.cache_data
    def load_cv_rep():
        if os.path.exists(CV_PATH):
            with open(CV_PATH) as f: return json.load(f)
        return {}

    cv_data = load_cv_rep()

    if not res:
        st.warning("⚠️ Hisobot ma'lumotlari topilmadi.")
        st.stop()

    rows = [{"Model":n,"Accuracy":m.get("accuracy",0),"F1":m.get("f1",0),
             "Precision":m.get("precision",0),"Recall":m.get("recall",0),
             "ROC-AUC":m.get("auc") or 0} for n,m in res.items()]
    df_r = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best = df_r.iloc[0]

    # Preview
    st.markdown("<div class='sh'>👁️ Hisobot ko'rinishi</div>", unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("Jami talabalar", f"{len(df):,}")
    p2.metric("Eng yaxshi model", best["Model"])
    p3.metric("Eng yuqori Accuracy", f"{best['Accuracy']:.2%}")
    p4.metric("Tahlil qilingan modellar", len(res))

    st.markdown("---")

    def build_html_report():
        import base64
        from datetime import datetime

        # Load figures as base64
        fig_html = ""
        if os.path.exists(FIGURES_DIR):
            for fn in sorted(os.listdir(FIGURES_DIR)):
                if fn.endswith(".png"):
                    with open(os.path.join(FIGURES_DIR, fn), "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    title = fn.replace(".png","").replace("_"," ").title()
                    fig_html += f"""
                    <div class="fig-wrap">
                      <h4>{title}</h4>
                      <img src="data:image/png;base64,{b64}" alt="{title}">
                    </div>"""

        # Model results table
        table_rows = ""
        for i, (_, row) in enumerate(df_r.iterrows()):
            highlight = ' style="background:#fffde7;"' if row["Model"]==best["Model"] else ""
            table_rows += f"""<tr{highlight}>
              <td>{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else str(i+1)}</td>
              <td><b>{row['Model']}</b></td>
              <td>{row['Accuracy']:.4f}</td>
              <td>{row['F1']:.4f}</td>
              <td>{row['Precision']:.4f}</td>
              <td>{row['Recall']:.4f}</td>
              <td>{row['ROC-AUC']:.4f}</td></tr>"""

        # CV table
        cv_rows = ""
        for name, v in cv_data.items():
            if v.get("mean") and not pd.isna(v.get("mean",float("nan"))):
                cv_rows += f"""<tr>
                  <td>{name}</td>
                  <td>{v['mean']:.4f}</td>
                  <td>{v['std']:.4f}</td>
                  <td>{v.get('min',0):.4f}</td>
                  <td>{v.get('max',0):.4f}</td></tr>"""

        counts = df["Target"].value_counts()
        counts_html = " · ".join([f"<b>{c}</b>: {v}" for c,v in counts.items()])

        html = f"""<!DOCTYPE html>
<html lang="uz">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>EduPredict AI — Diplom Loyihasi Hisoboti</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',Arial,sans-serif; background:#f5f7fa; color:#222; }}
    .header {{ background:linear-gradient(135deg,#7828ff,#0ea5e9); color:#fff; padding:48px 40px; }}
    .header h1 {{ font-size:2.2rem; font-weight:800; margin-bottom:8px; }}
    .header p {{ opacity:.85; font-size:1.05rem; }}
    .badge {{ display:inline-block; background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.4);
              border-radius:50px; padding:4px 14px; font-size:.8rem; margin-top:12px; margin-right:6px; }}
    .container {{ max-width:1200px; margin:0 auto; padding:32px 24px; }}
    .section {{ background:#fff; border-radius:16px; padding:28px; margin-bottom:24px;
                box-shadow:0 2px 16px rgba(0,0,0,.07); }}
    .section h2 {{ color:#7828ff; font-size:1.3rem; margin-bottom:18px; border-bottom:2px solid #f0f0f0; padding-bottom:10px; }}
    .stats {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:8px; }}
    .stat-box {{ flex:1; min-width:160px; background:linear-gradient(135deg,#7828ff15,#0ea5e915);
                 border:1px solid #7828ff30; border-radius:12px; padding:18px; text-align:center; }}
    .stat-box .val {{ font-size:1.8rem; font-weight:800; color:#7828ff; }}
    .stat-box .lbl {{ font-size:.78rem; color:#888; text-transform:uppercase; letter-spacing:.06em; margin-top:4px; }}
    table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
    th {{ background:#7828ff; color:#fff; padding:12px 14px; text-align:left; }}
    td {{ padding:10px 14px; border-bottom:1px solid #f0f0f0; }}
    tr:hover {{ background:#f8f5ff; }}
    .fig-wrap {{ margin:16px 0; text-align:center; }}
    .fig-wrap h4 {{ color:#555; margin-bottom:8px; font-size:.9rem; }}
    .fig-wrap img {{ max-width:100%; border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,.1); }}
    .figs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:20px; }}
    .footer {{ text-align:center; padding:24px; color:#aaa; font-size:.85rem; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>🎓 EduPredict AI — Diplom Loyihasi</h1>
    <p>Talabalar Dropout va Akademik Muvaffaqiyatini Bashorat Qilish</p>
    <span class="badge">📦 UCI Dataset · ID 697</span>
    <span class="badge">👥 {len(df):,} talaba</span>
    <span class="badge">🤖 {len(res)} ML model</span>
    <span class="badge">📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
  </div>
  <div class="container">

    <div class="section">
      <h2>📊 Dataset Umumiy Ko'rinishi</h2>
      <div class="stats">
        <div class="stat-box"><div class="val">{len(df):,}</div><div class="lbl">Jami talabalar</div></div>
        <div class="stat-box"><div class="val">{df.shape[1]-1}</div><div class="lbl">Xususiyatlar</div></div>
        <div class="stat-box"><div class="val">3</div><div class="lbl">Sinflar</div></div>
        <div class="stat-box"><div class="val">0</div><div class="lbl">Bo'sh qiymatlar</div></div>
      </div>
      <p><b>Sinflar:</b> {counts_html}</p>
    </div>

    <div class="section">
      <h2>🏆 Model Natijalari</h2>
      <div class="stats">
        <div class="stat-box"><div class="val">{best['Model']}</div><div class="lbl">Eng yaxshi model</div></div>
        <div class="stat-box"><div class="val">{best['Accuracy']:.2%}</div><div class="lbl">Eng yuqori Accuracy</div></div>
        <div class="stat-box"><div class="val">{best['ROC-AUC']:.4f}</div><div class="lbl">Eng yuqori AUC</div></div>
        <div class="stat-box"><div class="val">{best['F1']:.4f}</div><div class="lbl">Eng yuqori F1</div></div>
      </div>
      <table>
        <thead><tr><th>#</th><th>Model</th><th>Accuracy</th><th>F1</th>
          <th>Precision</th><th>Recall</th><th>ROC-AUC</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>

    {'<div class="section"><h2>📋 Cross-Validation Natijalari (5-Fold)</h2><table><thead><tr><th>Model</th><th>CV Mean</th><th>Std</th><th>Min</th><th>Max</th></tr></thead><tbody>' + cv_rows + '</tbody></table></div>' if cv_rows else ''}

    <div class="section">
      <h2>🖼️ Vizualizatsiya Grafiklar</h2>
      <div class="figs">{fig_html}</div>
    </div>

  </div>
  <div class="footer">
    EduPredict AI · Diplom loyihasi · {datetime.now().year} ·
    Machine Learning bilan talabalar natijasini bashorat qilish
  </div>
</body>
</html>"""
        return html.encode("utf-8")

    col1, col2, col3 = st.columns(3)

    if col1.button("🔨  HTML Hisobot yaratish", type="primary",
                   use_container_width=True, key="rep_gen"):
        with st.spinner("⏳ Hisobot yaratilmoqda..."):
            try:
                html_bytes = build_html_report()
                st.success(f"✅ Hisobot tayyor! ({len(html_bytes)//1024} KB)")
                st.download_button(
                    "⬇️  HTML hisobotni yuklab olish",
                    data=html_bytes,
                    file_name="edupredict_hisobot.html",
                    mime="text/html",
                    use_container_width=True,
                    key="rep_dl_html")
                st.info("💡 HTML faylni brauzerda oching va **Ctrl+P** → 'PDF sifatida saqlash' orqali PDF qiling.")
            except Exception as e:
                st.error(f"Xato: {e}")
                st.code(traceback.format_exc())

    col2.download_button(
        "⬇️  JSON yuklab olish",
        data=json.dumps(res, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="model_results.json",
        mime="application/json",
        use_container_width=True,
        key="rep_dl_json")

    col3.download_button(
        "⬇️  CSV jadval yuklab olish",
        data=to_csv(df_r),
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True,
        key="rep_dl_csv")


    st.markdown("---")
    st.markdown("<div class='sh'>📊 Hisobot tarkibi</div>", unsafe_allow_html=True)
    items = [
        ("✅","Dataset umumiy statistika","4,424 talaba, 36 xususiyat, 3 sinf"),
        ("✅","Barcha model natijalari",f"{len(res)} model — Accuracy, F1, Precision, Recall, AUC"),
        ("✅","Cross-Validation natijalari","5-fold CV mean ± std"),
        ("✅","Vizualizatsiya grafiklar",f"{len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')]) if os.path.exists(FIGURES_DIR) else 0} ta PNG grafik"),
        ("✅","Eng yaxshi model tafsiloti",f"{best['Model']} — {best['Accuracy']:.2%} accuracy"),
    ]
    for ico, title, desc in items:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:14px;padding:12px 16px;"
            f"background:rgba(255,255,255,.04);border-radius:10px;margin-bottom:8px;"
            f"border:1px solid rgba(139,92,246,.1);box-shadow:0 2px 8px rgba(139,92,246,.05);'>"
            f"<span style='font-size:1.2rem;'>{ico}</span>"
            f"<div><div style='font-weight:600;color:#E2E8F0;'>{title}</div>"
            f"<div style='font-size:.8rem;color:#475569;margin-top:2px;'>{desc}</div></div>"
            f"</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  14 · LOYIHA HAQIDA
# ════════════════════════════════════════════════════
elif "Loyiha Haqida" in page:
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">ℹ️ Diplom Loyihasi 2026</div>
      <div class="htitle">Loyiha Haqida</div>
      <div class="hsub">Machine Learning yordamida talabalar natijasini bashorat qilish tizimi — metodologiya, texnologiyalar va arxitektura</div>
    </div></div>""", unsafe_allow_html=True)

    # ── Maqsad
    st.markdown("<div class='sh'>🎯 Loyiha Maqsadi</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(139,92,246,.07);border:1px solid rgba(139,92,246,.15);
      border-radius:16px;padding:24px;margin-bottom:20px;
      box-shadow:0 4px 20px rgba(139,92,246,.06);'>
      <p style='font-size:1.05rem;line-height:1.8;color:#C4B5FD;'>
        Ushbu diplom loyihasi oliy ta'lim muassasasidagi talabalarning akademik natijalarini —
        <b style="color:#8B5CF6;">dropout (o'qishni tashlab ketish)</b>,
        <b style="color:#D97706;">enrolled (o'qishda davom etish)</b> va
        <b style="color:#16A34A;">graduate (muvaffaqiyatli bitirish)</b> —
        Machine Learning algoritmlari yordamida oldindan aniqlashga qaratilgan.
        Tizim 9 ta turli ML model qo'llaydi va ularni o'zaro taqqoslab, eng aniq bashoratni taqdim etadi.
      </p>
    </div>""", unsafe_allow_html=True)

    # ── Pipeline
    st.markdown("<div class='sh'>🔄 ML Pipeline</div>", unsafe_allow_html=True)
    steps = [
        ("1", "📦", "Ma'lumot yuklash", "UCI Dataset ID:697 · 4,424 talaba · 36 xususiyat", "#7828ff"),
        ("2", "🔧", "Preprocessing", "Ustun tozalash · Label Encoding · Feature Engineering (+7 yangi xususiyat)", "#0ea5e9"),
        ("3", "⚖️", "SMOTE Balans", "Sinf nomutanosibligi bartaraf · oversampling · 3 sinf tenglashtirish", "#00c853"),
        ("4", "✂️", "Train/Test Split", "80% train · 20% test · Stratified sampling", "#ff9f00"),
        ("5", "🤖", "Model O'qitish", "9 ta ML algoritm parallel o'qitish · Hyperparameter tuning", "#ff2d55"),
        ("6", "📊", "Baholash", "Accuracy · F1 · Precision · Recall · ROC-AUC · 5-Fold CV", "#e0aaff"),
        ("7", "🔬", "SHAP Tahlil", "Explainable AI · xususiyat ta'siri · global va lokal tushuntirish", "#7af5ff"),
        ("8", "🎯", "Bashorat", "Yangi talaba ma'lumotlari → real vaqt bashorat", "#ffd060"),
    ]
    cols2 = st.columns(4)
    for i, (num, ico, title, desc, clr) in enumerate(steps):
        cols2[i % 4].markdown(
            f"<div style='background:rgba(255,255,255,.04);border:1px solid rgba(139,92,246,.12);"
            f"border-top:3px solid {clr};border-radius:14px;padding:16px;margin-bottom:12px;"
            f"box-shadow:0 2px 12px rgba(139,92,246,.06);'>"
            f"<div style='font-size:.65rem;color:{clr};font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;'>QADAM {num}</div>"
            f"<div style='font-size:1.4rem;margin-bottom:6px;'>{ico}</div>"
            f"<div style='font-weight:700;color:#E2E8F0;font-size:.9rem;margin-bottom:4px;'>{title}</div>"
            f"<div style='font-size:.75rem;color:#475569;line-height:1.5;'>{desc}</div>"
            f"</div>", unsafe_allow_html=True)

    # ── Texnologiyalar
    st.markdown("<div class='sh'>🛠️ Texnologiyalar</div>", unsafe_allow_html=True)
    techs = [
        ("Python 3.x", "Asosiy dasturlash tili", "🐍", "#3776AB"),
        ("scikit-learn", "ML framework · preprocessing · evaluation", "🔬", "#F7931E"),
        ("XGBoost", "Gradient boosting · eng yuqori accuracy", "⚡", "#189AB4"),
        ("LightGBM", "Fast gradient boosting · leaf-wise", "💡", "#00B050"),
        ("SHAP", "Explainable AI · model tushuntirish", "🔍", "#FF6B6B"),
        ("SMOTE", "imbalanced-learn · class balancing", "⚖️", "#9B59B6"),
        ("Streamlit", "Web interfeys · interaktiv dashboard", "🌐", "#FF4B4B"),
        ("Plotly", "Interaktiv vizualizatsiya", "📊", "#636EFA"),
        ("Pandas / NumPy", "Ma'lumot tahlili va manipulyatsiya", "🐼", "#150458"),
    ]
    tc = st.columns(3)
    for i, (name, desc, ico, clr) in enumerate(techs):
        tc[i % 3].markdown(
            f"<div style='display:flex;align-items:center;gap:12px;background:rgba(255,255,255,.04);"
            f"border:1px solid rgba(139,92,246,.1);border-radius:12px;padding:14px;margin-bottom:10px;"
            f"border-left:3px solid {clr};box-shadow:0 2px 10px rgba(139,92,246,.05);'>"
            f"<span style='font-size:1.5rem;'>{ico}</span>"
            f"<div><div style='font-weight:700;color:#E2E8F0;font-size:.9rem;'>{name}</div>"
            f"<div style='font-size:.75rem;color:#475569;margin-top:2px;'>{desc}</div></div>"
            f"</div>", unsafe_allow_html=True)

    # ── Dataset
    st.markdown("<div class='sh'>📦 Dataset Tafsilotlari</div>", unsafe_allow_html=True)
    df_a = load_df()
    a1, a2, a3, a4 = st.columns(4)
    for col, ico, val, lbl in [
        (a1, "🏛️", "Portekiz", "Mamlakat"),
        (a2, "📅", "4,424 talaba", "Jami namunalar"),
        (a3, "🎓", "Polytechnic Institute", "Muassasa"),
        (a4, "📋", "UCI Repository ID:697", "Manba"),
    ]:
        col.markdown(
            f"<div class='stat'><span class='si'>{ico}</span>"
            f"<div class='sv' style='font-size:1.1rem;'>{val}</div><div class='sl'>{lbl}</div></div>",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature groups
    st.markdown("<div class='sh'>📐 Xususiyatlar Guruhlari</div>", unsafe_allow_html=True)
    fg_cols = st.columns(3)
    feature_groups_info = [
        ("👤 Demografik", ["Jins", "Yosh", "Millat", "Oilaviy holat", "Xorijiy", "Ko'chib kelgan", "Maxsus ehtiyoj"], "#7828ff"),
        ("💰 Ijtimoiy-Iqtisodiy", ["Stipendiya", "To'lov holati", "Qarzdorlik", "Ota-ona ta'limi", "Ota-ona kasbi"], "#0ea5e9"),
        ("📚 Akademik Fon", ["Kurs", "Ariza usuli", "Qabul bahosi", "Oldingi ta'lim", "Dars vaqti"], "#00c853"),
        ("📖 1-Semestr", ["Yozilgan", "Baholangan", "O'tilgan", "Baho", "Kreditlar", "Bahosiz"], "#ff9f00"),
        ("📖 2-Semestr", ["Yozilgan", "Baholangan", "O'tilgan", "Baho", "Kreditlar", "Bahosiz"], "#ff2d55"),
        ("🌍 Makroiqtisodiy", ["Ishsizlik darajasi", "Inflyatsiya", "YaIM o'sishi"], "#e0aaff"),
    ]
    for i, (grp_name, feats, clr) in enumerate(feature_groups_info):
        fg_cols[i % 3].markdown(
            f"<div style='background:rgba(255,255,255,.04);border:1px solid rgba(139,92,246,.1);"
            f"border-top:3px solid {clr};border-radius:14px;padding:16px;margin-bottom:12px;"
            f"box-shadow:0 2px 12px rgba(139,92,246,.05);'>"
            f"<div style='font-weight:700;color:{clr};margin-bottom:10px;font-size:.9rem;'>{grp_name}</div>"
            + "".join([f"<div style='font-size:.78rem;color:#475569;padding:3px 0;border-bottom:1px solid rgba(139,92,246,.06);'>• {f}</div>" for f in feats])
            + "</div>", unsafe_allow_html=True)

    # ── Modellar
    st.markdown("<div class='sh'>🤖 Qo'llanilgan ML Modellar</div>", unsafe_allow_html=True)
    res_a = load_res()
    models_info = [
        ("Logistic Regression", "Chiziqli, tezkor, sodda model — baseline", "📉", "Oson talqin etiladi"),
        ("Random Forest", "200 qaror daraxti, bagging, feature randomness", "🌲", "Yuqori barqarorlik"),
        ("Gradient Boosting", "Ketma-ket o'qitish, xatoliklarni tuzatish", "📈", "Yaxshi to'ldirishlar"),
        ("XGBoost", "Regularizatsiyali gradient boosting — champion", "⚡", "Eng yuqori accuracy"),
        ("LightGBM", "Leaf-wise o'sish, tez va samarali", "💡", "Katta datasetlarda tez"),
        ("SVM", "Kernel trick, yuqori o'lchamli makon", "🔵", "Chiziqsiz chegaralar"),
        ("KNN", "K yaqin qo'shni, masofaga asoslangan", "📍", "Oddiy, hisob-intensive"),
        ("Neural Network", "MLP 256→128→64, ReLU, Adam, Early Stopping", "🧠", "Chuqur o'rganish"),
        ("Ensemble", "XGBoost+LightGBM+RandomForest Soft Voting", "🗳️", "Eng barqaror natija"),
    ]
    mc = st.columns(3)
    for i, (mname, mdesc, mico, mtag) in enumerate(models_info):
        acc = res_a.get(mname, {}).get("accuracy", 0) if res_a else 0
        auc = res_a.get(mname, {}).get("auc") or 0 if res_a else 0
        mc[i % 3].markdown(
            f"<div style='background:rgba(255,255,255,.04);border:1px solid rgba(139,92,246,.1);"
            f"border-radius:14px;padding:16px;margin-bottom:10px;"
            f"box-shadow:0 2px 12px rgba(139,92,246,.05);'>"
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
            f"<span style='font-size:1.3rem;'>{mico}</span>"
            f"<div style='font-weight:700;color:#E2E8F0;font-size:.9rem;'>{mname}</div></div>"
            f"<div style='font-size:.75rem;color:#475569;margin-bottom:10px;'>{mdesc}</div>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-size:.78rem;background:rgba(139,92,246,.08);border-radius:50px;"
            f"padding:2px 10px;color:#8B5CF6;border:1px solid rgba(139,92,246,.18);'>{mtag}</span>"
            f"<span style='font-size:.82rem;font-weight:700;color:#16A34A;'>{acc:.1%}</span></div>"
            f"</div>", unsafe_allow_html=True)

    # ── Metodologiya
    st.markdown("<div class='sh'>📐 Baholash Metodologiyasi</div>", unsafe_allow_html=True)
    mc2 = st.columns(2)
    metrics_explain = [
        ("🎯 Accuracy", "To'g'ri bashorat qilingan talabalar ulushi", "Asosiy ko'rsatkich"),
        ("📊 F1 Score", "Precision va Recall harmonik o'rtachasi", "Nomutanosib sinflar uchun"),
        ("🔵 ROC-AUC", "Receiver Operating Characteristic maydoni", "0.9+ = ajoyib model"),
        ("📋 5-Fold CV", "Stratified K-Fold cross-validation", "Generalizatsiya qobiliyati"),
        ("🔬 SHAP", "SHapley Additive exPlanations (XAI)", "Har xususiyat ta'siri"),
        ("⚖️ SMOTE", "Synthetic Minority Oversampling Technique", "Sinf balansi"),
    ]
    for i, (mname, mdesc, mtag) in enumerate(metrics_explain):
        mc2[i % 2].markdown(
            f"<div style='display:flex;gap:14px;background:rgba(255,255,255,.04);"
            f"border:1px solid rgba(139,92,246,.1);border-radius:12px;"
            f"padding:14px;margin-bottom:10px;align-items:flex-start;"
            f"box-shadow:0 2px 10px rgba(139,92,246,.05);'>"
            f"<div style='font-size:1.3rem;flex-shrink:0;'>{mname.split()[0]}</div>"
            f"<div><div style='font-weight:700;color:#E2E8F0;font-size:.88rem;'>{' '.join(mname.split()[1:])}</div>"
            f"<div style='font-size:.76rem;color:#475569;margin-top:3px;'>{mdesc}</div>"
            f"<div style='font-size:.72rem;background:rgba(139,92,246,.08);border-radius:50px;"
            f"padding:1px 10px;color:#8B5CF6;margin-top:6px;display:inline-block;"
            f"border:1px solid rgba(139,92,246,.15);'>{mtag}</div>"
            f"</div></div>", unsafe_allow_html=True)

    # ── Natijalar xulosasi
    if res_a:
        st.markdown("<div class='sh'>🏆 Eng Yaxshi Natijalar Xulosasi</div>", unsafe_allow_html=True)
        rows_a = [{"Model":n,"Accuracy":m.get("accuracy",0),"F1":m.get("f1",0),
                   "Precision":m.get("precision",0),"Recall":m.get("recall",0),
                   "ROC-AUC":m.get("auc") or 0} for n,m in res_a.items()]
        df_a2 = pd.DataFrame(rows_a).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        best_a = df_a2.iloc[0]

        ab1, ab2, ab3, ab4 = st.columns(4)
        ab1.metric("🥇 Champion", best_a["Model"])
        ab2.metric("🎯 Max Accuracy", f"{best_a['Accuracy']:.2%}")
        ab3.metric("📊 Max F1", f"{best_a['F1']:.4f}")
        ab4.metric("🔵 Max AUC", f"{best_a['ROC-AUC']:.4f}")

        try:
            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(name="Accuracy", x=df_a2["Model"], y=df_a2["Accuracy"],
                marker_color="#7828ff", text=df_a2["Accuracy"].apply(lambda x:f"{x:.2%}"),
                textposition="outside", textfont=dict(color="#8B5CF6")))
            fig_a.add_trace(go.Bar(name="ROC-AUC", x=df_a2["Model"], y=df_a2["ROC-AUC"],
                marker_color="#0ea5e9", text=df_a2["ROC-AUC"].apply(lambda x:f"{x:.3f}"),
                textposition="outside", textfont=dict(color="#94A3B8")))
            apply_dark(fig_a, title="Barcha Modellar — Accuracy vs ROC-AUC",
                       barmode="group", height=380, legend=dict(font=dict(color="#A78BFA")))
            st.plotly_chart(fig_a, use_container_width=True, key="about_chart")
        except Exception as e:
            st.error(f"{e}")

    # ── Footer
    st.markdown("""
    <div style='text-align:center;margin-top:32px;padding:28px;
      background:rgba(139,92,246,.07);
      border:1px solid rgba(139,92,246,.15);border-radius:18px;
      box-shadow:0 4px 24px rgba(139,92,246,.08);'>
      <div style='font-size:2rem;margin-bottom:10px;filter:drop-shadow(0 4px 8px rgba(139,92,246,.25));'>🎓</div>
      <div style='font-family:"Space Grotesk",sans-serif;font-size:1.1rem;font-weight:700;
        background:linear-gradient(135deg,#6D28D9,#8B5CF6,#A78BFA);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        EduPredict AI — Diplom Loyihasi 2026
      </div>
      <div style='color:#64748B;font-size:.82rem;margin-top:8px;'>
        Machine Learning · XAI · Streamlit · Python · UCI Dataset
      </div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
