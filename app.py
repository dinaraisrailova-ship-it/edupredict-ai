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

# ══════════════════════════════════════ LANGUAGE
if "lang" not in st.session_state:
    st.session_state.lang = "UZ"

def EN():
    return st.session_state.lang == "EN"

def t(uz, en):
    return en if EN() else uz

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

    # ── Language toggle
    col_l, col_r = st.columns(2)
    if col_l.button("🇬🇧 English", use_container_width=True,
                    type="primary" if EN() else "secondary"):
        st.session_state.lang = "EN"; st.rerun()
    if col_r.button("🇺🇿 O'zbek", use_container_width=True,
                    type="primary" if not EN() else "secondary"):
        st.session_state.lang = "UZ"; st.rerun()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Pages",
        ["🏠  Dashboard",
         "📊  " + t("Ma'lumotlar","Data Explorer"),
         "🤖  " + t("Model Natijalari","Model Results"),
         "🔮  " + t("Bashorat","Predict"),
         "📤  " + t("Ommaviy Bashorat","Batch Predict"),
         "📉  " + t("Xavf Monitor","Risk Monitor"),
         "🔬  SHAP Analysis",
         "📋  Cross-Validation",
         "📄  " + t("Hisobot","Report")],
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
          <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:#334155;text-transform:uppercase;letter-spacing:.08em;font-weight:600;">Students</div>
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
    _dash_title = "Student Outcome<br>Prediction System" if EN() else "Talabalar Natijasini<br>Bashorat Qilish Tizimi"
    _dash_sub   = ("Predict student academic success, dropout risk and graduation probability "
                   "in real time using Machine Learning." if EN() else
                   "Machine Learning yordamida talabaning akademik muvaffaqiyatini, "
                   "dropout xavfini va bitirish ehtimolini real vaqtda oldindan aniqlash.")
    _dash_c4    = "📤 Batch Predict" if EN() else "📤 Batch Bashorat"
    _dash_c8    = "📄 HTML Report"   if EN() else "📄 HTML Hisobot"
    _dash_eye   = "Diploma project"  if EN() else "Diplom loyihasi"
    _dash_ml    = "Models"           if EN() else "Model"
    _dash_sm    = "Balance"          if EN() else "Balans"
    st.markdown(f"""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">🏆 {_dash_eye} &nbsp;·&nbsp; 2026</div>
      <div class="htitle">{_dash_title}</div>
      <div class="hsub">{_dash_sub}</div>
      <div class="chips">
        <span class="chip">🤖 9 ML {_dash_ml}</span>
        <span class="chip">🧬 SMOTE {_dash_sm}</span>
        <span class="chip">🔬 SHAP · XAI</span>
        <span class="chip">{_dash_c4}</span>
        <span class="chip">📉 Risk Monitor</span>
        <span class="chip">🗳️ Soft Voting Ensemble</span>
        <span class="chip">📋 Cross-Validation</span>
        <span class="chip">{_dash_c8}</span>
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
    @keyframes orb1{{0%,100%{{transform:translateY(0)}}33%{{transform:translateY(-8px)}}}}
    @keyframes orb2{{0%,100%{{transform:translateY(0)}}66%{{transform:translateY(-6px)}}}}
    @keyframes orb3{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-10px)}}}}
    </style>
    </div>""", unsafe_allow_html=True)

    df = load_df()
    res = load_res()
    best_name = max(res, key=lambda k: res[k].get("accuracy", 0)) if res else "XGBoost"
    best_acc  = res.get(best_name, {}).get("accuracy", 0)
    best_auc  = res.get(best_name, {}).get("auc") or 0

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, ico, val, lbl in [
        (c1,"👥",f"{len(df):,}",t("Jami talabalar","Total students")),
        (c2,"📐",f"{df.shape[1]-1}",t("Xususiyatlar","Features")),
        (c3,"🏆",best_name,t("Eng yaxshi model","Best model")),
        (c4,"🎯",f"{best_acc:.1%}",t("Eng yuqori Accuracy","Top Accuracy")),
        (c5,"🔵",f"{best_auc:.3f}",t("Eng yuqori AUC","Top AUC")),
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
            apply_dark(fig, title=t("Sinflar taqsimoti","Class distribution"), height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="dash_bar")
        except Exception as e:
            st.error(f"Bar chart: {e}")

    with cr:
        try:
            fig2 = go.Figure(go.Pie(
                labels=list(counts.index), values=list(counts.values),
                marker=dict(colors=[CCOL[c] for c in counts.index],
                            line=dict(color="rgba(0,0,0,.4)", width=3)),
                hole=.42, textfont=dict(size=13),
                hovertemplate=f"<b>%{{label}}</b><br>%{{value}} {t('talaba','students')}<br>%{{percent}}<extra></extra>",
            ))
            fig2.update_layout(**dark_fig(), title=t("Foiz taqsimoti","Percentage distribution"), height=340,
                               legend=dict(font=dict(color="#A78BFA")))
            st.plotly_chart(fig2, use_container_width=True, key="dash_pie")
        except Exception as e:
            st.error(f"Pie chart: {e}")

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
            apply_dark(fig3, title=t("Barcha modellar ko'rsatkichlari","All models performance"),
                       barmode="group", height=380, legend=dict(font=dict(color="#A78BFA")))
            st.plotly_chart(fig3, use_container_width=True, key="dash_models")
        except Exception as e:
            st.error(f"Model chart: {e}")



# ════════════════════════════════════════════════════
#  2 · DATA EXPLORER
# ════════════════════════════════════════════════════
elif "Data Explorer" in page or "Ma'lumotlar" in page:
    st.markdown(f"""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">📊 EDA</div>
      <div class="htitle">{"Data Analysis" if EN() else "Ma'lumotlar Tahlili"}</div>
      <div class="hsub">{"Explore dataset structure, distributions and correlations" if EN() else "Dataset tuzilmasi, taqsimot va korrelyatsiyalarni o'rganish"}</div>
    </div></div>""", unsafe_allow_html=True)

    df = load_df()
    _tabs = (["📋 Dataset","📊 Distribution","📈 Correlation","📦 Class Analysis","🖼️ Charts"] if EN() else
             ["📋 Dataset","📊 Taqsimot","📈 Korrelyatsiya","📦 Sinf Tahlili","🖼️ Grafiklar"])
    t1,t2,t3,t4,t5 = st.tabs(_tabs)

    with t1:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Rows" if EN() else "Qatorlar", f"{len(df):,}")
        m2.metric("Columns" if EN() else "Ustunlar", df.shape[1])
        m3.metric("Numeric cols" if EN() else "Raqamli ustun", df.select_dtypes(include=np.number).shape[1])
        m4.metric("Missing values" if EN() else "Bo'sh qiymat", df.isnull().sum().sum())
        n = st.slider("Rows to show" if EN() else "Ko'rsatiladigan qatorlar", 10, 300, 50, key="eda_slider")
        st.dataframe(df.head(n), use_container_width=True, height=380)

    with t2:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        col_sel = st.selectbox(t("Ustun tanlang:","Select column:"), num_cols, key="eda_col")
        try:
            fig = go.Figure()
            for cls, color in CCOL.items():
                sub = df[df["Target"] == cls][col_sel]
                if not sub.empty:
                    fig.add_trace(go.Histogram(x=sub, name=cls, marker_color=color,
                        opacity=.75, nbinsx=30))
            apply_dark(fig, barmode="overlay", title=f"{col_sel} — {t('taqsimot','distribution')}", height=360)
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
            fig.update_layout(**dark_fig(), title=t("Korrelyatsiya matritsasi","Correlation matrix"), height=680)
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
            apply_dark(fig2, title=t("Target bilan korrelyatsiya (Top 20)","Correlation with Target (Top 20)"), height=500)
            st.plotly_chart(fig2, use_container_width=True, key="eda_corr_bar")
        except Exception as e:
            st.error(f"Korrelyatsiya xatosi: {e}")

    with t4:
        feat_opts = [c for c in [
            "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
            "Curricular units 1st sem (approved)", "Age at enrollment", "Admission grade",
        ] if c in df.columns]
        feat4 = st.selectbox(t("Ko'rsatkich:","Feature:"), feat_opts, key="eda_feat4")
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
            _cls_dist = t("Sinf bo'yicha taqsimot","Distribution by class")
            fig2.update_layout(**dark_fig(),
                title=f"{feat4} — {_cls_dist}", height=320)
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
                st.info(t("📊 Grafiklar mavjud emas.","📊 No charts available."))


# ════════════════════════════════════════════════════
#  3 · MODEL RESULTS
# ════════════════════════════════════════════════════
elif "Model Results" in page or "Model Natijalari" in page:
    st.markdown(f"""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">🤖 {"ML Models" if EN() else "ML Modellar"}</div>
      <div class="htitle">{"Model Results" if EN() else "Model Natijalari"}</div>
      <div class="hsub">{"9 ML models trained, evaluated and compared" if EN() else "9 ta ML model o'qitildi, baholandi va taqqoslandi"}</div>
    </div></div>""", unsafe_allow_html=True)

    res = load_res()
    if not res:
        st.warning(t("⚠️ Ma'lumotlar topilmadi.","⚠️ No data found."))
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
    st.markdown(f"<div class='sh'>🔍 {t('Model tafsilotlari','Model Details')}</div>", unsafe_allow_html=True)
    sel = st.selectbox(t("Model tanlang:","Select model:"), MODELS, key="mr_sel")
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
        st.info(t("📊 Grafiklar mavjud emas.","📊 No charts available."))

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
            _pc_hdr = t("Sinf bo'yicha batafsil metrikalar","Detailed per-class metrics")
            st.markdown(f"<div class='sh'>📋 {_pc_hdr}</div>", unsafe_allow_html=True)
            pc_rep = pc_data[sel]
            rows_pc = []
            for cls in CLASSES:
                if cls in pc_rep and isinstance(pc_rep[cls], dict):
                    rows_pc.append({
                        t("Sinf","Class"): cls,
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
                f"⬇️  {sel} {t('modelini .pkl yuklab olish','model — download .pkl')}",
                data=f.read(),
                file_name=f"{safe}.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                key=f"mr_pkl_{safe}")


# ════════════════════════════════════════════════════
#  4 · PREDICT
# ════════════════════════════════════════════════════
elif ("Predict" in page or "Bashorat" in page) and "Batch" not in page and "Ommaviy" not in page:
    st.markdown(f"""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">🔮 {"AI Prediction" if EN() else "AI Bashorat"}</div>
      <div class="htitle">{"Predict Student Outcome" if EN() else "Talaba Holatini Aniqlash"}</div>
      <div class="hsub">{"Enter data — model instantly predicts Dropout · Enrolled · Graduate" if EN() else "Ma'lumotlarni kiriting — model darhol Dropout · Enrolled · Graduate bashorat qiladi"}</div>
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
    _demo_hdr = t("⚡ Demo namunalar — bir bosishda sinab ko'ring","⚡ Demo samples — try with one click")
    st.markdown(f"<div class='sh'>{_demo_hdr}</div>", unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    if dc1.button(t("🎓 A'lo talaba (Graduate)","🎓 Top student (Graduate)"), use_container_width=True, key="demo_grad"):
        for k, v in DEMOS["graduate"].items(): st.session_state[k] = v
        st.session_state["auto_predict"] = True
        st.rerun()
    if dc2.button(t("⚠️ Xavf ostida (Dropout)","⚠️ At-risk (Dropout)"), use_container_width=True, key="demo_drop"):
        for k, v in DEMOS["dropout"].items(): st.session_state[k] = v
        st.session_state["auto_predict"] = True
        st.rerun()
    if dc3.button(t("📚 O'qishda davom (Enrolled)","📚 Still Enrolled (Enrolled)"), use_container_width=True, key="demo_enrl"):
        for k, v in DEMOS["enrolled"].items(): st.session_state[k] = v
        st.session_state["auto_predict"] = True
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main form — 2 columns: left=personal+financial, right=academic
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(f"<div class='sh'>{t('👤 Shaxsiy va moliyaviy','👤 Personal & Financial')}</div>", unsafe_allow_html=True)
        age    = st.slider(t("🎂 Yosh (qabul paytida)","🎂 Age (at enrollment)"), 17, 70, 20, key="p_age")
        _yes   = t("✅ Ha","✅ Yes")
        _no    = t("❌ Yo'q","❌ No")
        gender = st.selectbox(t("⚧ Jins","⚧ Gender"), [0, 1],
                    format_func=lambda x: (t("👩 Ayol","👩 Female") if x == 0 else t("👨 Erkak","👨 Male")),
                    key="p_gender")
        tuition = st.selectbox(t("💳 To'lov muddatida to'langan","💳 Tuition fees up to date"), [1, 0],
                    format_func=lambda x: _yes if x == 1 else _no,
                    key="p_tuition")
        schol  = st.selectbox(t("🎓 Stipendiyant","🎓 Scholarship holder"), [0, 1],
                    format_func=lambda x: _no if x == 0 else _yes,
                    key="p_schol")
        debtor = st.selectbox(t("💸 Qarzdor","💸 Debtor"), [0, 1],
                    format_func=lambda x: _no if x == 0 else t("⚠️ Ha","⚠️ Yes"),
                    key="p_debtor")
        adg    = st.slider(t("📋 Qabul bahosi","📋 Admission grade"), 95.0, 190.0, 127.0, key="p_adg")

    with right:
        st.markdown(f"<div class='sh'>{t('📖 Akademik natijalar','📖 Academic results')}</div>", unsafe_allow_html=True)
        st.markdown(f"**{t('1-Semestr','Semester 1')}**")
        r1a, r1b, r1c = st.columns(3)
        s1en = r1a.slider(t("Yozilgan fan","Enrolled units"), 0, 26, 6, key="s1en")
        s1ap = r1b.slider(t("O'tilgan fan","Approved units"), 0, 26, 5, key="s1ap")
        s1gr = r1c.slider(t("O'rtacha baho","Avg grade"), 0.0, 20.0, 12.0, key="s1gr")

        st.markdown(f"**{t('2-Semestr','Semester 2')}**")
        r2a, r2b, r2c = st.columns(3)
        s2en = r2a.slider(t("Yozilgan fan","Enrolled units"), 0, 23, 6, key="s2en")
        s2ap = r2b.slider(t("O'tilgan fan","Approved units"), 0, 20, 5, key="s2ap")
        s2gr = r2c.slider(t("O'rtacha baho","Avg grade"), 0.0, 20.0, 12.0, key="s2gr")

    # ── Advanced settings expander
    st.markdown("<br>", unsafe_allow_html=True)
    _adv_lbl  = t("⚙️ Kengaytirilgan sozlamalar — qo'shimcha maydonlar (ixtiyoriy)","⚙️ Advanced settings — optional fields")
    _sh_soc   = t("📋 Ijtimoiy va oilaviy ma'lumotlar","📋 Social & family data")
    _sh_par   = t("👨‍👩‍👧 Ota-ona ma'lumotlari","👨‍👩‍👧 Parent data")
    _sh_adm   = t("🎓 Qabul va kurs ma'lumotlari","🎓 Admission & course data")
    _sh_sem   = t("📖 Semestr batafsil","📖 Semester detail")
    _sh_mac   = t("🌍 Makroiqtisodiy ko'rsatkichlar","🌍 Macroeconomic indicators")
    with st.expander(_adv_lbl, expanded=False):
        _yn  = lambda x: (_no if x==0 else _yes)
        st.markdown(f"<div class='sh'>{_sh_soc}</div>", unsafe_allow_html=True)
        adv1, adv2, adv3 = st.columns(3)
        _marital_uz = {1:"Yagona",2:"Turmush qurgan",3:"Keva",4:"Ajrashgan",5:"Birga yashovchi",6:"Qonuniy ajrashgan"}
        _marital_en = {1:"Single",2:"Married",3:"Widower",4:"Divorced",5:"Facto union",6:"Legally separated"}
        marital = adv1.selectbox(t("💍 Oilaviy holat","💍 Marital status"), [1,2,3,4,5,6],
            format_func=lambda x: (_marital_en if EN() else _marital_uz).get(x,str(x)),
            key="p_marital")
        displ   = adv2.selectbox(t("🏘️ Ko'chib kelgan","🏘️ Displaced"), [0,1],
            format_func=_yn, key="p_displ")
        spcn    = adv3.selectbox(t("♿ Maxsus ehtiyoj","♿ Special needs"), [0,1],
            format_func=_yn, key="p_spcn")

        adv4, adv5 = st.columns(2)
        intl    = adv4.selectbox(t("🌍 Xorijiy talaba","🌍 International student"), [0,1],
            format_func=_yn, key="p_intl")
        nation  = adv5.number_input(t("🗺️ Millat (kodi)","🗺️ Nationality (code)"), 1, 109, 1, key="p_nation")

        st.markdown(f"<div class='sh'>{_sh_par}</div>", unsafe_allow_html=True)
        oe1, oe2, oe3, oe4 = st.columns(4)
        mq = oe1.number_input(t("Ona ta'lim darajasi","Mother's education"), 0, 44, 19, key="p_mq")
        fq = oe2.number_input(t("Ota ta'lim darajasi","Father's education"), 0, 44, 22, key="p_fq")
        mo = oe3.number_input(t("Ona kasbi (kodi)","Mother's occupation"), 0, 194, 10, key="p_mo")
        fo = oe4.number_input(t("Ota kasbi (kodi)","Father's occupation"), 0, 194, 10, key="p_fo")

        st.markdown(f"<div class='sh'>{_sh_adm}</div>", unsafe_allow_html=True)
        qa1, qa2, qa3, qa4 = st.columns(4)
        amode = qa1.number_input(t("Ariza usuli (kodi)","Application mode (code)"), 1, 57, 1, key="p_amode")
        aord  = qa2.number_input(t("Ariza tartibi","Application order"), 0, 9, 1, key="p_aord")
        crs   = qa3.number_input(t("Kurs (kodi)","Course (code)"), 0, 9999, 9500, key="p_crs")
        att   = qa4.selectbox(t("Dars vaqti","Attendance time"), [1,0],
            format_func=lambda x: (t("🌞 Kunduzgi","🌞 Daytime") if x==1 else t("🌙 Kechki","🌙 Evening")), key="p_att")

        qa5, qa6 = st.columns(2)
        pq  = qa5.number_input(t("Oldingi ta'lim turi (kodi)","Previous qualification (code)"), 1, 43, 1, key="p_pq")
        pqg = qa6.slider(t("Oldingi ta'lim bahosi","Previous qualification grade"), 0.0, 200.0, 130.0, key="p_pqg")

        st.markdown(f"<div class='sh'>{_sh_sem}</div>", unsafe_allow_html=True)
        sem_a, sem_b = st.columns(2)
        with sem_a:
            st.caption(t("1-Semestr","Semester 1"))
            s1cr = st.number_input(t("Kredit (1-sem)","Credit (sem 1)"), 0, 20, 0, key="s1cr")
            s1ev = st.number_input(t("Baholangan (1-sem)","Evaluated (sem 1)"), 0, 45, s1en, key="s1ev")
            s1ne = st.number_input(t("Bahosiz (1-sem)","No eval (sem 1)"), 0, 19, 0, key="s1ne")
        with sem_b:
            st.caption(t("2-Semestr","Semester 2"))
            s2cr = st.number_input(t("Kredit (2-sem)","Credit (sem 2)"), 0, 19, 0, key="s2cr")
            s2ev = st.number_input(t("Baholangan (2-sem)","Evaluated (sem 2)"), 0, 45, s2en, key="s2ev")
            s2ne = st.number_input(t("Bahosiz (2-sem)","No eval (sem 2)"), 0, 12, 0, key="s2ne")

        st.markdown(f"<div class='sh'>{_sh_mac}</div>", unsafe_allow_html=True)
        me1, me2, me3 = st.columns(3)
        unemp = me1.slider(t("📉 Ishsizlik darajasi (%)","📉 Unemployment rate (%)"), 7.0, 17.0, 11.5, 0.1, key="p_unemp")
        infl  = me2.slider(t("📈 Inflyatsiya darajasi (%)","📈 Inflation rate (%)"), -0.8, 3.3, 1.2, 0.1, key="p_infl")
        gdp   = me3.slider(t("💹 YaIM o'sishi (%)","💹 GDP growth (%)"), -4.1, 3.5, 0.0, 0.1, key="p_gdp")

    # ── Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    _, bc, _ = st.columns([1, 2, 1])
    go_pred   = bc.button(t("🔮  Bashorat Qilish","🔮  Predict"), type="primary",
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
            b_txt = {
                "Dropout":  t("⚠️ DROPOUT XAVFI","⚠️ DROPOUT RISK"),
                "Enrolled": t("📚 O'QISHDA DAVOM ETADI","📚 STILL ENROLLED"),
                "Graduate": t("🎓 MUVAFFAQIYATLI BITIRADI","🎓 WILL GRADUATE"),
            }[label]

            st.markdown("---")
            _, rc, _ = st.columns([1, 2, 1])
            with rc:
                st.markdown(
                    f"<div style='text-align:center;padding:16px 0;'>"
                    f"<div style='font-size:.8rem;color:#8B5CF6;margin-bottom:12px;"
                    f"letter-spacing:.08em;text-transform:uppercase;font-weight:700;'>"
                    f"{t('Bashorat natijasi','Prediction result')} · {sel_m}</div>"
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
                "Dropout":  t("⚠️ **Xavf darajasi: YUQORI** — Talabaga shoshilinch akademik va moliyaviy yordam kerak.",
                              "⚠️ **Risk level: HIGH** — Student needs urgent academic and financial support."),
                "Enrolled": t("📚 **Holat: KUZATUVDA** — Talaba o'qishda. Motivatsiya va doimiy monitoring tavsiya etiladi.",
                              "📚 **Status: MONITORING** — Student is enrolled. Motivation and regular monitoring recommended."),
                "Graduate": t("🎓 **Holat: A'LO** — Talaba muvaffaqiyatli bitirish yo'lida. Mavjud sharoitlarni saqlang.",
                              "🎓 **Status: EXCELLENT** — Student is on track to graduate. Maintain current conditions."),
            }
            st.info(tip[label])

            # ── Barcha modellar taqqoslash
            st.markdown("---")
            st.markdown(f"<div class='sh'>🔬 {t('Barcha modellar taqqoslash','All models comparison')}</div>", unsafe_allow_html=True)
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
                    title=t("Barcha modellar — Dropout · Enrolled · Graduate ehtimoli","All models — Dropout · Enrolled · Graduate probability"),
                    barmode="group", height=360,
                    legend=dict(font=dict(color="#A78BFA")),
                    yaxis_tickformat=".0%", yaxis_range=[0, 1.15])
                st.plotly_chart(fig_cmp, use_container_width=True, key="pred_all_models")

                # Ovoz berish
                votes = {}
                for r in all_results:
                    votes[r["label"]] = votes.get(r["label"], 0) + 1
                winner = max(votes, key=votes.get)
                _vote_word = t("ovoz","votes")
                vote_html = " &nbsp;·&nbsp; ".join(
                    [f"<b style='color:{CCOL[c]}'>{c}: {v} {_vote_word}</b>" for c, v in sorted(votes.items(), key=lambda x: -x[1])]
                )
                _vote_hdr = t("🗳️ UMUMIY OVOZ NATIJASI","🗳️ OVERALL VOTE RESULT")
                st.markdown(
                    f"<div style='text-align:center;padding:14px;background:rgba(0,212,255,.04);"
                    f"border:1px solid rgba(0,212,255,.16);border-radius:12px;margin-top:8px;'>"
                    f"<div style='font-size:.72rem;color:#64748B;letter-spacing:.1em;text-transform:uppercase;"
                    f"font-family:JetBrains Mono,monospace;margin-bottom:6px;'>{_vote_hdr}</div>"
                    f"<div style='font-size:.95rem;'>{vote_html}</div>"
                    f"</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Bashorat xatosi: {e}")
            st.code(traceback.format_exc())


# ════════════════════════════════════════════════════
# ════════════════════════════════════════════════════
#  5 · BATCH PREDICT
# ════════════════════════════════════════════════════
elif "Batch" in page or "Ommaviy" in page:
    st.markdown(f"""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">📤 {"Batch Prediction" if EN() else "Ommaviy Bashorat"}</div>
      <div class="htitle">Batch Prediction</div>
      <div class="hsub">{"Upload CSV — predict all students at once" if EN() else "CSV faylni yuklang — barcha talabalar uchun bir vaqtda bashorat"}</div>
    </div></div>""", unsafe_allow_html=True)

    sel_m = st.selectbox("🤖 Model:", MODELS, key="batch_model")
    model = load_mdl(sel_m)
    if not model:
        st.error(t("❌ Model topilmadi.","❌ Model not found."))
        st.stop()

    df_full = load_df()
    tmpl = df_full.drop(columns=["Target"]).head(5)
    st.download_button(t("⬇️  Namuna CSV (shablon) yuklab olish","⬇️  Download sample CSV (template)"),
        data=to_csv(tmpl), file_name="template.csv", mime="text/csv",
        use_container_width=True, key="batch_tmpl")

    uploaded = st.file_uploader(t("📂  CSV fayl yuklang:","📂  Upload CSV file:"), type=["csv"], key="batch_upload")

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ {t('Yuklandi','Uploaded')}: **{df_up.shape[0]}** {t('qator','rows')} · **{df_up.shape[1]}** {t('ustun','cols')}")

            # ── CSV ustunlarini tekshirish
            required_cols = set(load_df().drop(columns=["Target"]).columns)
            uploaded_cols = set(df_up.columns)
            missing_cols  = required_cols - uploaded_cols
            extra_cols    = uploaded_cols - required_cols
            if missing_cols:
                st.error(f"❌ {t('CSV da quyidagi ustunlar yetishmayapti','Missing columns in CSV')} ({len(missing_cols)}):\n`{', '.join(sorted(missing_cols))}`")
                st.info(t("💡 To'g'ri formatda shablon yuklab olish uchun yuqoridagi tugmani bosing.","💡 Click the button above to download a correctly formatted template."))
                st.stop()
            if extra_cols:
                _extra_msg = t("Qo'shimcha ustunlar topildi (e'tiborga olinmaydi)","Extra columns found (ignored)")
                st.warning(f"⚠️ {_extra_msg}: `{', '.join(sorted(extra_cols))}`")
                df_up = df_up[list(required_cols)]

            null_count = df_up.isnull().sum().sum()
            if null_count > 0:
                _null_msg = t("ta bo'sh qiymat topildi — o'rtacha bilan to'ldiriladi.","missing values found — filled with median.")
                st.warning(f"⚠️ {null_count} {_null_msg}")
                df_up = df_up.fillna(df_up.median(numeric_only=True))

            st.dataframe(df_up.head(8), use_container_width=True)

            if st.button(t("🚀  Barchasi uchun bashorat","🚀  Predict all"), type="primary",
                         use_container_width=True, key="batch_run"):
                with st.spinner(t("⏳ Bashorat qilinmoqda...","⏳ Predicting...")):
                    try:
                        preds, probs = run_predict(model, df_up)
                        df_res = df_up.copy()
                        _pred_col = t("Bashorat","Prediction")
                        df_res[_pred_col]    = [CLASSES[p] for p in preds]
                        df_res["Dropout_%"]  = [round(float(p[0])*100,1) for p in probs]
                        df_res["Enrolled_%"] = [round(float(p[1])*100,1) for p in probs]
                        df_res["Graduate_%"] = [round(float(p[2])*100,1) for p in probs]

                        cnt = pd.Series([CLASSES[p] for p in preds]).value_counts()
                        mc1,mc2,mc3 = st.columns(3)
                        mc1.metric(t("⚠️ Dropout xavfi","⚠️ Dropout risk"), int(cnt.get("Dropout",0)))
                        mc2.metric(t("📚 O'qishda","📚 Enrolled"),           int(cnt.get("Enrolled",0)))
                        mc3.metric(t("🎓 Bitiradi","🎓 Will graduate"),       int(cnt.get("Graduate",0)))

                        fig = go.Figure(go.Pie(
                            labels=list(cnt.index), values=list(cnt.values),
                            marker=dict(colors=[CCOL[c] for c in cnt.index],
                                        line=dict(color="rgba(0,0,0,.3)",width=2)),
                            hole=.4, textfont=dict(size=13)))
                        fig.update_layout(**dark_fig(), title=t("Bashorat taqsimoti","Prediction distribution"), height=360,
                                          legend=dict(font=dict(color="#A78BFA")))
                        st.plotly_chart(fig, use_container_width=True, key="batch_pie")

                        st.dataframe(
                            df_res[[_pred_col,"Dropout_%","Enrolled_%","Graduate_%"]
                                   + list(df_up.columns[:4])].head(60),
                            use_container_width=True)

                        st.download_button(
                            t("⬇️  Barcha natijalarni CSV yuklab olish","⬇️  Download all results as CSV"),
                            data=to_csv(df_res),
                            file_name="batch_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="batch_dl_csv")

                    except Exception as e:
                        st.error(f"{t('Bashorat xatosi','Prediction error')}: {e}")
                        st.code(traceback.format_exc())
        except Exception as e:
            _ferr = t("Fayl o'qish xatosi","File read error")
            st.error(f"{_ferr}: {e}")

    st.markdown("---")
    st.markdown(f"<div class='sh'>📊 {t('Mavjud dataset ustida test','Test on existing dataset')}</div>", unsafe_allow_html=True)
    if st.button(t("🔄  To'liq dataset ustida bashorat qilish","🔄  Predict on full dataset"), type="secondary",
                 use_container_width=True, key="batch_full"):
        with st.spinner(t("⏳ Bashorat qilinmoqda...","⏳ Predicting...")):
            try:
                p2, pr2 = run_predict(model, df_full.drop(columns=["Target"]))
                df_t = df_full.copy()
                _pred_col2 = t("Bashorat","Prediction")
                df_t[_pred_col2] = [CLASSES[p] for p in p2]
                acc = (df_t["Target"] == df_t[_pred_col2]).mean()
                st.success(f"✅ {t('Umumiy aniqlik','Overall accuracy')}: **{acc:.2%}** ({int(acc*len(df_t))}/{len(df_t)})")
                st.dataframe(df_t[["Target",_pred_col2]].head(50), use_container_width=True)
                st.download_button(
                    t("⬇️  To'liq natijalar CSV","⬇️  Full results CSV"),
                    data=to_csv(df_t[["Target",_pred_col2]]),
                    file_name="full_dataset_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="batch_full_dl")
            except Exception as e:
                st.error(f"{t('Xato','Error')}: {e}")
                st.code(traceback.format_exc())


# ════════════════════════════════════════════════════
#  6 · RISK MONITOR
# ════════════════════════════════════════════════════
elif "Risk" in page or "Xavf" in page:
    st.markdown(f"""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">📉 {"Risk Monitor" if EN() else "Xavf Monitori"}</div>
      <div class="htitle">Risk Dashboard</div>
      <div class="hsub">{"Identify and filter high-risk dropout students" if EN() else "Dropout xavfi bo'lgan talabalarni aniqlash va filtrlash"}</div>
    </div></div>""", unsafe_allow_html=True)

    sel_m = st.selectbox("🤖 Model:", MODELS, key="risk_model")
    model = load_mdl(sel_m)
    if not model:
        st.error(t("❌ Model topilmadi.","❌ Model not found."))
        st.stop()

    df_full = load_df()
    with st.spinner(t("⏳ Barcha talabalar uchun bashorat qilinmoqda...","⏳ Predicting for all students...")):
        try:
            preds, probs = run_predict(model, df_full.drop(columns=["Target"]))
        except Exception as e:
            st.error(f"{t('Bashorat xatosi','Prediction error')}: {e}")
            st.stop()

    df_r = df_full.copy()
    df_r["Bashorat"]  = [CLASSES[p] for p in preds]
    df_r["Dropout_p"] = [float(p[0]) for p in probs]
    df_r["Togri"]     = df_r["Target"] == df_r["Bashorat"]

    st.markdown(f"<div class='sh'>🎛️ {t('Filtrlar','Filters')}</div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    thresh     = f1.slider(t("Dropout chegara ehtimoli","Dropout threshold probability"), 0.0, 1.0, 0.5, 0.05, key="risk_thresh")
    g_sel      = f2.multiselect(t("Jins","Gender"), [0,1], default=[0,1], key="risk_gender",
                                format_func=lambda x: t("Ayol","Female") if x==0 else t("Erkak","Male"))
    age_rng    = f3.slider(t("Yosh diapazoni","Age range"), 17, 70, (17,70), key="risk_age")

    df_f = df_r[df_r["Gender"].isin(g_sel) & df_r["Age at enrollment"].between(*age_rng)]
    high = df_f[df_f["Dropout_p"] >= thresh].sort_values("Dropout_p", ascending=False)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric(t("🔍 Filtrlangan","🔍 Filtered"), len(df_f))
    m2.metric(t("⚠️ Yuqori xavf","⚠️ High risk"), len(high))
    m3.metric(t("📊 Xavf ulushi","📊 Risk share"), f"{len(high)/max(len(df_f),1)*100:.1f}%")
    m4.metric(t("✅ Aniqlik","✅ Accuracy"), f"{df_f['Togri'].mean():.2%}")

    try:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure(go.Histogram(x=df_f["Dropout_p"], nbinsx=30,
                marker=dict(color="#ff2d55", opacity=.8)))
            fig.add_vline(x=thresh, line_color="#ffd200", line_width=2.5,
                          annotation_text=f"{t('Chegara','Threshold')}: {thresh}",
                          annotation_font_color="#ffd200")
            apply_dark(fig, title=t("Dropout ehtimoli taqsimoti","Dropout probability distribution"), height=320)
            st.plotly_chart(fig, use_container_width=True, key="risk_hist")

        with col_b:
            cnt2 = df_f["Bashorat"].value_counts()
            fig2 = go.Figure(go.Pie(
                labels=list(cnt2.index), values=list(cnt2.values),
                marker=dict(colors=[CCOL[c] for c in cnt2.index],
                            line=dict(color="rgba(0,0,0,.3)",width=2)),
                hole=.4, textfont=dict(size=13)))
            fig2.update_layout(**dark_fig(), title=t("Bashorat taqsimoti","Prediction distribution"), height=320,
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
        apply_dark(fig3, title=t("Yosh guruhlari bo'yicha Dropout ehtimoli","Dropout probability by age group"),
                   height=320, yaxis_range=[0,1])
        st.plotly_chart(fig3, use_container_width=True, key="risk_age")

        fig4 = px.scatter(
            df_f.sample(min(400, len(df_f)), random_state=42),
            x="Age at enrollment", y="Dropout_p", color="Target",
            color_discrete_map=CCOL, opacity=.65,
            title=t("Yosh va Dropout ehtimoli","Age vs Dropout probability"))
        fig4.update_layout(**dark_fig(), height=340, legend=dict(font=dict(color="#A78BFA")))
        dark_axes(fig4)
        st.plotly_chart(fig4, use_container_width=True, key="risk_scatter")
    except Exception as e:
        st.error(f"Chart xatosi: {e}")

    st.markdown(f"<div class='sh'>⚠️ {t('Yuqori xavfdagi talabalar','High-risk students')} — {len(high)}</div>",
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
    st.markdown(f"""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">🔬 XAI — SHAP</div>
      <div class="htitle">{"SHAP Analysis" if EN() else "SHAP Tahlili"}</div>
      <div class="hsub">{"Explain how each feature impacts predictions — Explainable AI" if EN() else "Har bir xususiyatning bashoratga ta'sirini tushuntirish — Explainable AI"}</div>
    </div></div>""", unsafe_allow_html=True)

    SHAP_CACHE = os.path.join(BASE, "reports", "shap_cache.pkl")

    @st.cache_data
    def load_shap_cache():
        if not os.path.exists(SHAP_CACHE): return {}
        with open(SHAP_CACHE, "rb") as f: return _pkl.load(f)

    cache = load_shap_cache()
    available = [m for m in ["XGBoost","LightGBM","Random Forest","Gradient Boosting"] if m in cache]

    if not available:
        st.warning(t("⏳ SHAP cache topilmadi. `python precompute_shap.py` ni ishga tushiring.","⏳ SHAP cache not found. Run `python precompute_shap.py`."))
        st.stop()

    sel_m = st.selectbox("🤖 Model:", available, key="shap_model")
    data  = cache[sel_m]
    sv    = data["shap_values"]
    feat  = data["feature_names"]
    X_s   = data["X_sample"]
    n_samples = X_s.shape[0]

    # ── 1. Global importance
    _shap_g = t("🌍 Global xususiyat ta'siri (Top 20)","🌍 Global feature importance (Top 20)")
    st.markdown(f"<div class='sh'>{_shap_g}</div>", unsafe_allow_html=True)
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
    _shap_title = t(f"Top 20 SHAP Global Ta'sir — {sel_m} ({n_samples} namuna)",f"Top 20 SHAP Global Impact — {sel_m} ({n_samples} samples)")
    apply_dark(fig, title=_shap_title,
               height=560, yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True, key="shap_global")

    # ── 2. Per-class breakdown
    if isinstance(sv, list) and len(sv) == 3:
        _shap_cls = t("🎯 Sinf bo'yicha xususiyat ta'siri (Top 10)","🎯 Per-class feature importance (Top 10)")
        st.markdown(f"<div class='sh'>{_shap_cls}</div>", unsafe_allow_html=True)
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

    # ── 3. Feature importance table (top 8)
    st.markdown("---")
    _shap_tbl = t("📊 Xususiyat muhimlik jadvali","📊 Feature importance table")
    st.markdown(f"<div class='sh'>{_shap_tbl}</div>", unsafe_allow_html=True)
    top8_feats = top_feats[:8]
    top8_vals  = top_vals[:8]
    _col_feat  = t("Xususiyat","Feature")
    _col_shap  = t("O'rtacha |SHAP|","Mean |SHAP|")
    _col_lvl   = t("Ta'sir darajasi","Impact level")
    _lvls_uz   = ["Past", "O'rta", "Yuqori", "Juda yuqori"]
    _lvls_en   = ["Low", "Medium", "High", "Very high"]
    df_imp = pd.DataFrame({_col_feat: top8_feats, _col_shap: top8_vals})
    df_imp[_col_lvl] = pd.cut(top8_vals,
        bins=[0, 0.05, 0.15, 0.30, 1.0],
        labels=_lvls_en if EN() else _lvls_uz)
    st.dataframe(df_imp.style.background_gradient(subset=[_col_shap], cmap="Reds"),
                 use_container_width=True, hide_index=True)

    # ── 4. Individual student SHAP
    st.markdown("---")
    _shap_ind = t("🔍 Alohida talaba SHAP tahlili","🔍 Individual student SHAP analysis")
    st.markdown(f"<div class='sh'>{_shap_ind}</div>", unsafe_allow_html=True)

    model_obj = load_mdl(sel_m)
    if model_obj:
        idx_s = st.slider(t("Talaba indeksi (0 – namuna ichidan)","Student index (0 – within sample)"), 0, n_samples - 1, 0,
                          key="shap_idx")
        X_row = pd.DataFrame([X_s[idx_s]], columns=feat)
        pred_l = CLASSES[model_obj.predict(X_row)[0]]
        pred_p = model_obj.predict_proba(X_row)[0]

        ic1, ic2, ic3 = st.columns(3)
        ic1.metric(t("Bashorat","Prediction"), pred_l)
        ic2.metric(t("Ishonch","Confidence"), f"{max(pred_p):.1%}")
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
        _shap_ind_title = t(f"Talaba #{idx_s} — '{pred_l}' sinfi uchun SHAP", f"Student #{idx_s} — SHAP for '{pred_l}' class")
        apply_dark(fig3, title=_shap_ind_title,
                   height=480, yaxis_autorange="reversed")
        st.plotly_chart(fig3, use_container_width=True, key="shap_ind")

    st.success(f"✅ {t('SHAP tahlili tayyor','SHAP analysis complete')} — {sel_m} · {n_samples} {t('namuna asosida','samples')}")


# ════════════════════════════════════════════════════
#  11 · CROSS-VALIDATION
# ════════════════════════════════════════════════════
elif "Cross-Validation" in page:
    st.markdown(f"""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">📋 Cross-Validation</div>
      <div class="htitle">{"CV Results" if EN() else "CV Natijalari"}</div>
      <div class="hsub">{"5-fold stratified CV — measuring true generalization of models" if EN() else "5-fold stratified CV — modellarning haqiqiy generalizatsiya qobiliyatini o'lchash"}</div>
    </div></div>""", unsafe_allow_html=True)

    @st.cache_data
    def load_cv():
        if os.path.exists(CV_PATH):
            with open(CV_PATH) as f: return json.load(f)
        return {}

    cv = load_cv()
    if not cv:
        st.warning(t("⚠️ CV natijalari topilmadi.","⚠️ CV results not found."))
        st.stop()

    valid_cv = {k:v for k,v in cv.items() if v.get("mean") and not pd.isna(v["mean"])}
    df_cv = pd.DataFrame([
        {"Model": k, "CV Mean": v["mean"], "Std": v["std"],
         "Min": v["min"], "Max": v["max"]}
        for k, v in valid_cv.items()
    ]).sort_values("CV Mean", ascending=False).reset_index(drop=True)

    best_cv = df_cv.iloc[0]
    c1,c2,c3 = st.columns(3)
    c1.metric(t("🥇 Eng yaxshi CV","🥇 Best CV"), best_cv["Model"])
    c2.metric("📊 CV Mean", f"{best_cv['CV Mean']:.4f}")
    c3.metric(t("📉 Std (barqarorlik)","📉 Std (stability)"), f"±{best_cv['Std']:.4f}")

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
        apply_dark(fig2, title=t("CV fold natijalari taqsimoti (5 fold)","CV fold results distribution (5 fold)"), height=400)
        st.plotly_chart(fig2, use_container_width=True, key="cv_box")
    except Exception as e:
        st.error(f"{e}")

    _cv_tbl = t("To'liq CV jadvali","Full CV table")
    st.markdown(f"<div class='sh'>📋 {_cv_tbl}</div>", unsafe_allow_html=True)
    st.dataframe(
        df_cv.style.format({c:"{:.4f}" for c in ["CV Mean","Std","Min","Max"]})
        .background_gradient(subset=["CV Mean"], cmap="Greens")
        .bar(subset=["Std"], color="#ff2d55", vmin=0, vmax=0.05),
        use_container_width=True)


    with st.expander(t("ℹ️ Cross-Validation nima?","ℹ️ What is Cross-Validation?")):
        if EN():
            st.markdown("""
        **5-Fold Stratified Cross-Validation** process:

        1. Train set is split into 5 equal folds (stratified — class ratio preserved)
        2. Each iteration uses 4 folds for training, 1 fold for validation
        3. 5 accuracy scores are collected and averaged
        4. This gives a more reliable estimate than simple test-set accuracy

        **Std (standard deviation)** small = stable model, large = unstable.
        """)
        else:
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
elif "Hisobot" in page or "Report" in page:
    st.markdown(f"""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">📄 {"Auto Report" if EN() else "Avtomatik Hisobot"}</div>
      <div class="htitle">{"Report Generation" if EN() else "Hisobot Generatsiya"}</div>
      <div class="hsub">{"Download project results in HTML or JSON format" if EN() else "Loyiha natijalarini HTML yoki JSON formatda yuklab oling"}</div>
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
        st.warning(t("⚠️ Hisobot ma'lumotlari topilmadi.","⚠️ Report data not found."))
        st.stop()

    rows = [{"Model":n,"Accuracy":m.get("accuracy",0),"F1":m.get("f1",0),
             "Precision":m.get("precision",0),"Recall":m.get("recall",0),
             "ROC-AUC":m.get("auc") or 0} for n,m in res.items()]
    df_r = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best = df_r.iloc[0]

    _rep_prev = t("Hisobot ko'rinishi","Report preview")
    st.markdown(f"<div class='sh'>👁️ {_rep_prev}</div>", unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    p1.metric(t("Jami talabalar","Total students"), f"{len(df):,}")
    p2.metric(t("Eng yaxshi model","Best model"), best["Model"])
    p3.metric(t("Eng yuqori Accuracy","Top Accuracy"), f"{best['Accuracy']:.2%}")
    p4.metric(t("Tahlil qilingan modellar","Models analyzed"), len(res))

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

    if col1.button(t("🔨  HTML Hisobot yaratish","🔨  Generate HTML Report"), type="primary",
                   use_container_width=True, key="rep_gen"):
        with st.spinner(t("⏳ Hisobot yaratilmoqda...","⏳ Generating report...")):
            try:
                html_bytes = build_html_report()
                st.success(f"✅ {t('Hisobot tayyor!','Report ready!')} ({len(html_bytes)//1024} KB)")
                st.download_button(
                    t("⬇️  HTML hisobotni yuklab olish","⬇️  Download HTML report"),
                    data=html_bytes,
                    file_name="edupredict_report.html",
                    mime="text/html",
                    use_container_width=True,
                    key="rep_dl_html")
                st.info(t("💡 HTML faylni brauzerda oching va **Ctrl+P** → 'PDF sifatida saqlash' orqali PDF qiling.","💡 Open the HTML file in a browser and press **Ctrl+P** → 'Save as PDF'."))
            except Exception as e:
                st.error(f"{t('Xato','Error')}: {e}")
                st.code(traceback.format_exc())

    col2.download_button(
        t("⬇️  JSON yuklab olish","⬇️  Download JSON"),
        data=json.dumps(res, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="model_results.json",
        mime="application/json",
        use_container_width=True,
        key="rep_dl_json")

    col3.download_button(
        t("⬇️  CSV jadval yuklab olish","⬇️  Download CSV table"),
        data=to_csv(df_r),
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True,
        key="rep_dl_csv")

    st.markdown("---")
    _rep_cont = t("Hisobot tarkibi","Report contents")
    st.markdown(f"<div class='sh'>📊 {_rep_cont}</div>", unsafe_allow_html=True)
    _n_figs = len([f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]) if os.path.exists(FIGURES_DIR) else 0
    items = [
        ("✅", t("Dataset umumiy statistika","Dataset overview"),        t(f"4,424 talaba, 36 xususiyat, 3 sinf",f"4,424 students, 36 features, 3 classes")),
        ("✅", t("Barcha model natijalari","All model results"),          f"{len(res)} model — Accuracy, F1, Precision, Recall, AUC"),
        ("✅", t("Cross-Validation natijalari","Cross-Validation results"), "5-fold CV mean ± std"),
        ("✅", t("Vizualizatsiya grafiklar","Visualization charts"),      f"{_n_figs} {t('ta PNG grafik','PNG charts')}"),
        ("✅", t("Eng yaxshi model tafsiloti","Best model detail"),       f"{best['Model']} — {best['Accuracy']:.2%} accuracy"),
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
elif "About" in page or "Loyiha" in page:
    st.markdown("""
    <div class="hero"><div class="hero-content">
      <div class="eyebrow">DIPLOMA PROJECT · DIPLOM LOYIHASI · 2026</div>
      <div class="htitle">EduPredict <span style="color:#7C3AED;">AI</span></div>
      <div class="hsub">
        <span style="color:#00D4FF;">Student Dropout Prediction System</span>
        <span style="color:#475569;"> · </span>
        <span style="color:#A78BFA;">Talabalar Natijasini Bashorat Qilish Tizimi</span>
      </div>
    </div></div>""", unsafe_allow_html=True)

    # ── Language tabs
    lang = st.radio("", ["🇬🇧  English", "🇺🇿  O'zbek"], horizontal=True, label_visibility="collapsed")
    EN = "English" in lang

    # ── Goal
    goal_title = "🎯 Project Goal" if EN else "🎯 Loyiha Maqsadi"
    goal_text_en = """This diploma project is designed to predict students' academic outcomes —
        <b style="color:#F87171;">Dropout</b>,
        <b style="color:#FBBF24;">Enrolled</b>, and
        <b style="color:#34D399;">Graduate</b> —
        using Machine Learning algorithms. The system employs 9 different ML models,
        compares them against each other, and delivers the most accurate prediction possible."""
    goal_text_uz = """Ushbu diplom loyihasi oliy ta'lim muassasasidagi talabalarning akademik natijalarini —
        <b style="color:#F87171;">Dropout (o'qishni tashlab ketish)</b>,
        <b style="color:#FBBF24;">Enrolled (o'qishda davom etish)</b> va
        <b style="color:#34D399;">Graduate (muvaffaqiyatli bitirish)</b> —
        Machine Learning algoritmlari yordamida oldindan aniqlashga qaratilgan.
        Tizim 9 ta turli ML model qo'llaydi va ularni o'zaro taqqoslab, eng aniq bashoratni taqdim etadi."""
    st.markdown(f"<div class='sh'>{goal_title}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.15);
      border-left:4px solid #00D4FF;border-radius:16px;padding:24px;margin-bottom:20px;'>
      <p style='font-size:1.05rem;line-height:1.9;color:#C4B5FD;'>{goal_text_en if EN else goal_text_uz}</p>
    </div>""", unsafe_allow_html=True)

    # ── Pipeline
    pipeline_title = "🔄 ML Pipeline" if EN else "🔄 ML Bosqichlari"
    st.markdown(f"<div class='sh'>{pipeline_title}</div>", unsafe_allow_html=True)
    steps = [
        ("1", "📦", "Data Loading" if EN else "Ma'lumot Yuklash",       "UCI ID:697 · 4,424 students · 36 features" if EN else "UCI ID:697 · 4,424 talaba · 36 xususiyat", "#7828ff"),
        ("2", "🔧", "Preprocessing",                                      "Cleaning · Label Encoding · +7 engineered features" if EN else "Tozalash · Label Encoding · +7 yangi xususiyat", "#0ea5e9"),
        ("3", "⚖️", "SMOTE Balance" if EN else "SMOTE Balans",           "Class imbalance fix · 3-class oversampling" if EN else "Sinf nomutanosibligi bartaraf · 3 sinf", "#00c853"),
        ("4", "✂️", "Train/Test Split",                                   "80% train · 20% test · Stratified" , "#ff9f00"),
        ("5", "🤖", "Model Training" if EN else "Model O'qitish",         "9 ML algorithms · Hyperparameter tuning" if EN else "9 ta ML algoritm · Hyperparameter tuning", "#ff2d55"),
        ("6", "📊", "Evaluation" if EN else "Baholash",                   "Accuracy · F1 · Precision · Recall · ROC-AUC · 5-CV", "#e0aaff"),
        ("7", "🔬", "SHAP Analysis" if EN else "SHAP Tahlil",             "Explainable AI · global & local feature impact" if EN else "Explainable AI · xususiyat ta'siri", "#7af5ff"),
        ("8", "🎯", "Prediction" if EN else "Bashorat",                   "New student data → real-time prediction" if EN else "Yangi talaba → real vaqt bashorat", "#ffd060"),
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

    # ── Technologies
    tech_title = "🛠️ Technologies Used" if EN else "🛠️ Texnologiyalar"
    st.markdown(f"<div class='sh'>{tech_title}</div>", unsafe_allow_html=True)
    techs = [
        ("Python 3.x",     "Core programming language" if EN else "Asosiy dasturlash tili",                "🐍", "#3776AB"),
        ("scikit-learn",   "ML framework · preprocessing · evaluation",                                     "🔬", "#F7931E"),
        ("XGBoost",        "Regularized gradient boosting · top accuracy" if EN else "Eng yuqori accuracy", "⚡", "#189AB4"),
        ("LightGBM",       "Fast leaf-wise gradient boosting" if EN else "Tez va samarali boosting",        "💡", "#00B050"),
        ("SHAP",           "Explainable AI · model interpretation" if EN else "Model tushuntirish",         "🔍", "#FF6B6B"),
        ("SMOTE",          "Synthetic minority oversampling · class balance" if EN else "Sinf balansi",     "⚖️", "#9B59B6"),
        ("Streamlit",      "Web interface · interactive dashboard" if EN else "Web interfeys · dashboard",  "🌐", "#FF4B4B"),
        ("Plotly",         "Interactive visualizations" if EN else "Interaktiv vizualizatsiya",             "📊", "#636EFA"),
        ("Pandas / NumPy", "Data analysis & manipulation" if EN else "Ma'lumot tahlili",                   "🐼", "#150458"),
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
    dataset_title = "📦 Dataset Details" if EN else "📦 Dataset Tafsilotlari"
    st.markdown(f"<div class='sh'>{dataset_title}</div>", unsafe_allow_html=True)
    df_a = load_df()
    a1, a2, a3, a4 = st.columns(4)
    for col, ico, val, lbl in [
        (a1, "🏛️", "Portugal",          "Country" if EN else "Mamlakat"),
        (a2, "📅", "4,424 students" if EN else "4,424 talaba", "Total samples" if EN else "Jami namunalar"),
        (a3, "🎓", "Polytechnic Institute", "Institution" if EN else "Muassasa"),
        (a4, "📋", "UCI Repository ID:697", "Source" if EN else "Manba"),
    ]:
        col.markdown(
            f"<div class='stat'><span class='si'>{ico}</span>"
            f"<div class='sv' style='font-size:1.1rem;'>{val}</div><div class='sl'>{lbl}</div></div>",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature groups
    fg_title = "📐 Feature Groups" if EN else "📐 Xususiyatlar Guruhlari"
    st.markdown(f"<div class='sh'>{fg_title}</div>", unsafe_allow_html=True)
    fg_cols = st.columns(3)
    feature_groups_info = [
        ("👤 Demographic" if EN else "👤 Demografik",
         ["Gender","Age","Nationality","Marital status","International","Displaced","Special needs"] if EN else
         ["Jins","Yosh","Millat","Oilaviy holat","Xorijiy","Ko'chib kelgan","Maxsus ehtiyoj"], "#7828ff"),
        ("💰 Socioeconomic" if EN else "💰 Ijtimoiy-Iqtisodiy",
         ["Scholarship","Tuition fees","Debtor","Parent education","Parent occupation"] if EN else
         ["Stipendiya","To'lov holati","Qarzdorlik","Ota-ona ta'limi","Ota-ona kasbi"], "#0ea5e9"),
        ("📚 Academic Background" if EN else "📚 Akademik Fon",
         ["Course","Application mode","Admission grade","Prior qualification","Attendance"] if EN else
         ["Kurs","Ariza usuli","Qabul bahosi","Oldingi ta'lim","Dars vaqti"], "#00c853"),
        ("📖 Semester 1" if EN else "📖 1-Semestr",
         ["Enrolled","Evaluated","Approved","Grade","Credits","Without eval."] if EN else
         ["Yozilgan","Baholangan","O'tilgan","Baho","Kreditlar","Bahosiz"], "#ff9f00"),
        ("📖 Semester 2" if EN else "📖 2-Semestr",
         ["Enrolled","Evaluated","Approved","Grade","Credits","Without eval."] if EN else
         ["Yozilgan","Baholangan","O'tilgan","Baho","Kreditlar","Bahosiz"], "#ff2d55"),
        ("🌍 Macroeconomic" if EN else "🌍 Makroiqtisodiy",
         ["Unemployment rate","Inflation rate","GDP growth"] if EN else
         ["Ishsizlik darajasi","Inflyatsiya","YaIM o'sishi"], "#e0aaff"),
    ]
    for i, (grp_name, feats, clr) in enumerate(feature_groups_info):
        fg_cols[i % 3].markdown(
            f"<div style='background:rgba(255,255,255,.04);border:1px solid rgba(139,92,246,.1);"
            f"border-top:3px solid {clr};border-radius:14px;padding:16px;margin-bottom:12px;"
            f"box-shadow:0 2px 12px rgba(139,92,246,.05);'>"
            f"<div style='font-weight:700;color:{clr};margin-bottom:10px;font-size:.9rem;'>{grp_name}</div>"
            + "".join([f"<div style='font-size:.78rem;color:#475569;padding:3px 0;border-bottom:1px solid rgba(139,92,246,.06);'>• {f}</div>" for f in feats])
            + "</div>", unsafe_allow_html=True)

    # ── Models
    models_title = "🤖 ML Models Used" if EN else "🤖 Qo'llanilgan ML Modellar"
    st.markdown(f"<div class='sh'>{models_title}</div>", unsafe_allow_html=True)
    res_a = load_res()
    models_info = [
        ("Logistic Regression", "Linear baseline · fast · interpretable" if EN else "Chiziqli, tezkor, sodda model", "📉", "Baseline" if EN else "Oson talqin"),
        ("Random Forest",       "200 trees · bagging · feature randomness" if EN else "200 qaror daraxti, bagging", "🌲", "Stable" if EN else "Barqaror"),
        ("Gradient Boosting",   "Sequential learning · error correction" if EN else "Ketma-ket o'qitish", "📈", "Robust" if EN else "Mustahkam"),
        ("XGBoost",             "Regularized gradient boosting · top model" if EN else "Regularizatsiyali boosting", "⚡", "Champion"),
        ("LightGBM",            "Leaf-wise growth · fast & efficient" if EN else "Leaf-wise, tez va samarali", "💡", "Fast" if EN else "Tez"),
        ("SVM",                 "Kernel trick · high-dim space" if EN else "Kernel trick, yuqori o'lchamli", "🔵", "Nonlinear" if EN else "Chiziqsiz"),
        ("KNN",                 "K-nearest neighbors · distance-based" if EN else "K yaqin qo'shni", "📍", "Simple" if EN else "Oddiy"),
        ("Neural Network",      "MLP 256→128→64 · ReLU · Adam · EarlyStopping", "🧠", "Deep Learning"),
        ("Ensemble",            "XGBoost+LightGBM+RF Soft Voting" if EN else "Soft Voting birlashmasi", "🗳️", "Most stable" if EN else "Eng barqaror"),
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

    # ── Methodology
    method_title = "📐 Evaluation Methodology" if EN else "📐 Baholash Metodologiyasi"
    st.markdown(f"<div class='sh'>{method_title}</div>", unsafe_allow_html=True)
    mc2 = st.columns(2)
    metrics_explain = [
        ("🎯 Accuracy",  "Share of correctly predicted students" if EN else "To'g'ri bashorat qilingan talabalar ulushi",   "Main metric" if EN else "Asosiy ko'rsatkich"),
        ("📊 F1 Score",  "Harmonic mean of Precision & Recall" if EN else "Precision va Recall harmonik o'rtachasi",         "Imbalanced classes" if EN else "Nomutanosib sinflar"),
        ("🔵 ROC-AUC",   "Area under the ROC curve" if EN else "Receiver Operating Characteristic maydoni",                 "0.9+ = excellent" if EN else "0.9+ = ajoyib"),
        ("📋 5-Fold CV", "Stratified K-Fold cross-validation" if EN else "Stratified K-Fold cross-validation",              "Generalization" if EN else "Umumlashtirish"),
        ("🔬 SHAP",      "SHapley Additive exPlanations (XAI)" if EN else "SHapley Additive exPlanations (XAI)",            "Feature impact" if EN else "Xususiyat ta'siri"),
        ("⚖️ SMOTE",    "Synthetic Minority Oversampling Technique" if EN else "Synthetic Minority Oversampling Technique",  "Class balance" if EN else "Sinf balansi"),
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

    # ── Results
    if res_a:
        results_title = "🏆 Best Results Summary" if EN else "🏆 Eng Yaxshi Natijalar Xulosasi"
        st.markdown(f"<div class='sh'>{results_title}</div>", unsafe_allow_html=True)
        rows_a = [{"Model":n,"Accuracy":m.get("accuracy",0),"F1":m.get("f1",0),
                   "Precision":m.get("precision",0),"Recall":m.get("recall",0),
                   "ROC-AUC":m.get("auc") or 0} for n,m in res_a.items()]
        df_a2 = pd.DataFrame(rows_a).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        best_a = df_a2.iloc[0]

        ab1, ab2, ab3, ab4 = st.columns(4)
        ab1.metric("🥇 Champion", best_a["Model"])
        ab2.metric("🎯 Accuracy", f"{best_a['Accuracy']:.2%}")
        ab3.metric("📊 F1 Score", f"{best_a['F1']:.4f}")
        ab4.metric("🔵 ROC-AUC", f"{best_a['ROC-AUC']:.4f}")

        try:
            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(name="Accuracy", x=df_a2["Model"], y=df_a2["Accuracy"],
                marker_color="#7828ff", text=df_a2["Accuracy"].apply(lambda x:f"{x:.2%}"),
                textposition="outside", textfont=dict(color="#8B5CF6")))
            fig_a.add_trace(go.Bar(name="ROC-AUC", x=df_a2["Model"], y=df_a2["ROC-AUC"],
                marker_color="#0ea5e9", text=df_a2["ROC-AUC"].apply(lambda x:f"{x:.3f}"),
                textposition="outside", textfont=dict(color="#94A3B8")))
            apply_dark(fig_a, title="All Models — Accuracy vs ROC-AUC" if EN else "Barcha Modellar — Accuracy vs ROC-AUC",
                       barmode="group", height=380, legend=dict(font=dict(color="#A78BFA")))
            st.plotly_chart(fig_a, use_container_width=True, key="about_chart")
        except Exception as e:
            st.error(f"{e}")

    # ── Footer
    footer_line1 = "EduPredict AI — Diploma Project 2026" if EN else "EduPredict AI — Diplom Loyihasi 2026"
    footer_line2 = "Machine Learning · Explainable AI · Streamlit · Python · UCI Dataset" if EN else "Machine Learning · Tushuntiriluvchi AI · Streamlit · Python · UCI Ma'lumotlari"
    st.markdown(f"""
    <div style='text-align:center;margin-top:40px;padding:36px;
      background:linear-gradient(135deg,rgba(0,212,255,.05),rgba(124,58,237,.07));
      border:1px solid rgba(0,212,255,.2);border-radius:20px;
      box-shadow:0 4px 40px rgba(0,212,255,.08);'>
      <div style='font-size:2.5rem;margin-bottom:12px;
        filter:drop-shadow(0 0 12px rgba(0,212,255,.5));'>🎓</div>
      <div style='font-family:"Space Grotesk",sans-serif;font-size:1.25rem;font-weight:800;
        background:linear-gradient(135deg,#00D4FF,#7C3AED,#A78BFA);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        text-shadow:none;letter-spacing:0.02em;'>
        {footer_line1}
      </div>
      <div style='color:#475569;font-size:.85rem;margin-top:10px;letter-spacing:0.05em;'>
        {footer_line2}
      </div>
      <div style='margin-top:16px;display:flex;justify-content:center;gap:12px;flex-wrap:wrap;'>
        <span style='background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.25);
          color:#00D4FF;border-radius:50px;padding:4px 14px;font-size:.78rem;font-weight:600;'>9 ML Models</span>
        <span style='background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.25);
          color:#A78BFA;border-radius:50px;padding:4px 14px;font-size:.78rem;font-weight:600;'>4,424 Students</span>
        <span style='background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);
          color:#22C55E;border-radius:50px;padding:4px 14px;font-size:.78rem;font-weight:600;'>XAI / SHAP</span>
        <span style='background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.25);
          color:#FBBF24;border-radius:50px;padding:4px 14px;font-size:.78rem;font-weight:600;'>UCI Dataset</span>
      </div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
