#!/usr/bin/env python3
"""
feat/feat_000.mqh ~ feat/feat_699.mqh を生成する。
各ファイルは CalcFeat_NNN(buf, idx, total_size, WarmData) を定義し、
Python features.py と同じ計算でスカラー値を返す。

メインEAから:
  #include "feat/feat_000.mqh"
  ...
  double val = CalcFeat_000(close[], high[], low[], open[], needed, i);
"""
import sys, os, json, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'fx-ea5'))
import features as fx_feat

BASE_COLS = fx_feat.BASE_FEATURE_COLS   # list of 230 names
N_GROUPS  = len(BASE_COLS)
OUT_DIR   = Path(__file__).parent / 'feat'
OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 特徴量名 → MQL5計算コードのマッピング
# 各エントリは関数本体の "return ..." 部分だけを文字列で返す
# 引数: close[], high[], low[], open[], vol[], needed (配列サイズ), i (最新=0)
# ──────────────────────────────────────────────────────────────────────────────

def mql_sma(p):
    return f"""
   double sum = 0.0;
   int cnt = 0;
   for(int k = i; k < i + {p} && k < needed; k++) {{ sum += close[k]; cnt++; }}
   return (cnt > 0) ? sum / cnt : close[i];"""

def mql_ema(p):
    # alpha = 2/(span+1), init at oldest, walk forward
    return f"""
   double alpha = 2.0 / ({p}.0 + 1.0);
   double em = close[needed-1];
   for(int k = needed-2; k >= i; k--)
      em = alpha * close[k] + (1.0 - alpha) * em;
   return em;"""

def mql_wma(p):
    return f"""
   double num = 0.0, den = 0.0;
   for(int k = 0; k < {p} && i+k < needed; k++) {{
      double w = (double)({p} - k);
      num += close[i+k] * w;
      den += w;
   }}
   return (den > 0) ? num/den : close[i];"""

def mql_hma(p):
    half = p // 2
    sq   = int(p**0.5)
    return f"""
   // HMA({p}) = WMA(2*WMA({half})-WMA({p}), sqrt({p}))
   double num_h=0,den_h=0, num_f=0,den_f=0;
   for(int k=0;k<{half}&&i+k<needed;k++){{double w={half}-k; num_h+=close[i+k]*w; den_h+=w;}}
   for(int k=0;k<{p}&&i+k<needed;k++){{double w={p}-k; num_f+=close[i+k]*w; den_f+=w;}}
   double wma_half=(den_h>0)?num_h/den_h:close[i];
   double wma_full=(den_f>0)?num_f/den_f:close[i];
   double diff_val = 2.0*wma_half - wma_full;
   // WMA(diff, {sq}): need series of diff at i+0..i+{sq-1}
   double num_s=0,den_s=0;
   for(int k=0;k<{sq}&&i+k<needed;k++){{
      double nh=0,dh=0,nf=0,df=0;
      for(int j=0;j<{half}&&i+k+j<needed;j++){{double w={half}-j;nh+=close[i+k+j]*w;dh+=w;}}
      for(int j=0;j<{p}&&i+k+j<needed;j++){{double w={p}-j;nf+=close[i+k+j]*w;df+=w;}}
      double dv=2.0*(dh>0?nh/dh:close[i+k])-(df>0?nf/df:close[i+k]);
      double w={sq}-k; num_s+=dv*w; den_s+=w;
   }}
   return (den_s>0)?num_s/den_s:diff_val;"""

def mql_macd(fast, slow, sig, part):
    # part: 'line','sig','hist'
    code = f"""
   // MACD({fast},{slow},{sig}) {part}
   double af={2.0/(fast+1)}, as_={2.0/(slow+1)}, asig={2.0/(sig+1)};
   double ef=close[needed-1], es=close[needed-1];
   for(int k=needed-2;k>=i;k--){{ef=af*close[k]+(1-af)*ef; es=as_*close[k]+(1-as_)*es;}}
   double ml=ef-es;
   // sig: need series of macd_line values back sig bars
   double sig_val=ml;
   // approx: run EMA of macd_line over full series
   double ef2=close[needed-1], es2=close[needed-1], prev_ml=0, sv=0;
   int first=1;
   for(int k=needed-2;k>=i;k--){{
      ef2=af*close[k]+(1-af)*ef2; es2=as_*close[k]+(1-as_)*es2;
      double cur_ml=ef2-es2;
      if(first){{sv=cur_ml;first=0;}} else sv=asig*cur_ml+(1-asig)*sv;
   }}
   sig_val=sv;"""
    if part == 'line':
        return code + "\n   return ml;"
    elif part == 'sig':
        return code + "\n   return sig_val;"
    else:  # hist
        return code + "\n   return ml - sig_val;"

def mql_rsi(period):
    return f"""
   double alpha=1.0/{period}.0;
   double ag=0,al=0;
   ag=al=0;
   int inited=0;
   for(int k=needed-2;k>=i;k--){{
      double d=close[k]-close[k+1];
      double g=(d>0)?d:0.0, l=(d<0)?-d:0.0;
      if(!inited){{ag=g;al=l;inited=1;}} else {{ag=alpha*g+(1-alpha)*ag; al=alpha*l+(1-alpha)*al;}}
   }}
   double rs=ag/(al+1e-9);
   return (100.0 - 100.0/(1.0+rs))/100.0;"""

def mql_stoch(k_period, d_period, part):
    # part= 'k' or 'd'
    code = f"""
   // Stoch({k_period},{d_period}) {part}
   // fast K
   double hi=high[i],lo_val=low[i];
   for(int k=1;k<{k_period}&&i+k<needed;k++){{hi=MathMax(hi,high[i+k]);lo_val=MathMin(lo_val,low[i+k]);}}
   double fk=(close[i]-lo_val)/(hi-lo_val+1e-9);"""
    if part == 'k':
        # smooth K by SMA(d_period) 
        code += f"""
   // smooth
   double sk=0; int sc=0;
   for(int j=0;j<{d_period}&&i+j<needed;j++){{
      double h2=high[i+j],l2=low[i+j];
      for(int k2=1;k2<{k_period}&&i+j+k2<needed;k2++){{h2=MathMax(h2,high[i+j+k2]);l2=MathMin(l2,low[i+j+k2]);}}
      sk+=(close[i+j]-l2)/(h2-l2+1e-9); sc++;
   }}
   return (sc>0)?sk/sc:fk;"""
    else:  # d = SMA(slow_k, d_period)
        code += f"""
   double sk[{d_period}+1]; int sc2=0;
   for(int j=0;j<{d_period}&&i+j<needed;j++){{
      double h2=high[i+j],l2=low[i+j];
      for(int k2=1;k2<{k_period}&&i+j+k2<needed;k2++){{h2=MathMax(h2,high[i+j+k2]);l2=MathMin(l2,low[i+j+k2]);}}
      double fk2=(close[i+j]-l2)/(h2-l2+1e-9);
      // smooth k
      double ssk=0; int ssc=0;
      for(int j2=0;j2<{d_period}&&i+j+j2<needed;j2++){{
         double h3=high[i+j+j2],l3=low[i+j+j2];
         for(int k3=1;k3<{k_period}&&i+j+j2+k3<needed;k3++){{h3=MathMax(h3,high[i+j+j2+k3]);l3=MathMin(l3,low[i+j+j2+k3]);}}
         ssk+=(close[i+j+j2]-l3)/(h3-l3+1e-9); ssc++;
      }}
      sk[j]=(ssc>0)?ssk/ssc:fk2; sc2++;
   }}
   double sd=0; for(int j=0;j<sc2;j++) sd+=sk[j];
   return (sc2>0)?sd/sc2:fk;"""
    return code

def mql_bb(period, part):
    # part: 'pos' or 'width'
    return f"""
   double s=0; int c2=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{s+=close[i+k];c2++;}}
   double m=(c2>0)?s/c2:close[i];
   double sq=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{double d=close[i+k]-m;sq+=d*d;}}
   double std=MathSqrt((c2>0)?sq/c2:0);
   double up=m+2*std, dn=m-2*std;
   double rng=up-dn;
   {"return (rng>1e-9)?(close[i]-dn)/rng:0.5;" if part=='pos' else "return (m>1e-9)?rng/m:0.0;"}"""

def mql_atr(period):
    return f"""
   double alpha=1.0/{period}.0;
   double tr0=high[needed-1]-low[needed-1], rm=tr0;
   for(int k=needed-2;k>=i;k--){{
      double hl=high[k]-low[k];
      double hpc=MathAbs(high[k]-close[k+1]);
      double lpc=MathAbs(low[k]-close[k+1]);
      double tr=MathMax(hl,MathMax(hpc,lpc));
      rm=alpha*tr+(1-alpha)*rm;
   }}
   return rm;"""

def mql_rma_series(period, what):
    """Generic RMA for a series computed from OHLC"""
    if what == 'tr':
        return f"""
   double alpha=1.0/{period}.0;
   double rm=high[needed-1]-low[needed-1];
   for(int k=needed-2;k>=i;k--){{
      double hl=high[k]-low[k];
      double hpc=MathAbs(high[k]-close[k+1]);
      double lpc=MathAbs(low[k]-close[k+1]);
      double tr=MathMax(hl,MathMax(hpc,lpc));
      rm=alpha*tr+(1-alpha)*rm;
   }}
   return rm;"""

def mql_adx(period):
    return f"""
   double alpha=1.0/{period}.0;
   double rm_tr=high[needed-1]-low[needed-1];
   double rm_pdm=0,rm_ndm=0;
   for(int k=needed-2;k>=i;k--){{
      double hl=high[k]-low[k];
      double hpc=MathAbs(high[k]-close[k+1]);
      double lpc=MathAbs(low[k]-close[k+1]);
      double tr=MathMax(hl,MathMax(hpc,lpc));
      rm_tr=alpha*tr+(1-alpha)*rm_tr;
      double up_m=high[k]-high[k+1];
      double dn_m=low[k+1]-low[k];
      double pdm=(up_m>dn_m&&up_m>0)?up_m:0.0;
      double ndm=(dn_m>up_m&&dn_m>0)?dn_m:0.0;
      rm_pdm=alpha*pdm+(1-alpha)*rm_pdm;
      rm_ndm=alpha*ndm+(1-alpha)*rm_ndm;
   }}
   double pdi=100*rm_pdm/(rm_tr+1e-9);
   double ndi=100*rm_ndm/(rm_tr+1e-9);
   double dx=100*MathAbs(pdi-ndi)/(pdi+ndi+1e-9);
   // ADX = RMA(dx) -- approximate with single pass value
   return dx/{period}.0;  // simplified approximation"""

def mql_pdi(period):
    return f"""
   double alpha=1.0/{period}.0;
   double rm_tr=high[needed-1]-low[needed-1];
   double rm_pdm=0;
   for(int k=needed-2;k>=i;k--){{
      double hl=high[k]-low[k];
      double hpc=MathAbs(high[k]-close[k+1]);
      double lpc=MathAbs(low[k]-close[k+1]);
      double tr=MathMax(hl,MathMax(hpc,lpc));
      rm_tr=alpha*tr+(1-alpha)*rm_tr;
      double up_m=high[k]-high[k+1];
      double dn_m=low[k+1]-low[k];
      double pdm=(up_m>dn_m&&up_m>0)?up_m:0.0;
      rm_pdm=alpha*pdm+(1-alpha)*rm_pdm;
   }}
   return 100*rm_pdm/(rm_tr+1e-9)/100.0;"""

def mql_ndi(period):
    return f"""
   double alpha=1.0/{period}.0;
   double rm_tr=high[needed-1]-low[needed-1];
   double rm_ndm=0;
   for(int k=needed-2;k>=i;k--){{
      double hl=high[k]-low[k];
      double hpc=MathAbs(high[k]-close[k+1]);
      double lpc=MathAbs(low[k]-close[k+1]);
      double tr=MathMax(hl,MathMax(hpc,lpc));
      rm_tr=alpha*tr+(1-alpha)*rm_tr;
      double up_m=high[k]-high[k+1];
      double dn_m=low[k+1]-low[k];
      double ndm=(dn_m>up_m&&dn_m>0)?dn_m:0.0;
      rm_ndm=alpha*ndm+(1-alpha)*rm_ndm;
   }}
   return 100*rm_ndm/(rm_tr+1e-9)/100.0;"""

def mql_cci(period):
    return f"""
   double tp=(high[i]+low[i]+close[i])/3.0;
   double s=0; int c2={period};
   for(int k=0;k<{period}&&i+k<needed;k++) s+=(high[i+k]+low[i+k]+close[i+k])/3.0;
   double m=s/c2;
   double mad=0;
   for(int k=0;k<{period}&&i+k<needed;k++) mad+=MathAbs((high[i+k]+low[i+k]+close[i+k])/3.0-m);
   mad/=c2;
   return (tp-m)/(0.015*mad+1e-9)/100.0;"""

def mql_wr(period):
    return f"""
   double hi_p=high[i],lo_p=low[i];
   for(int k=1;k<{period}&&i+k<needed;k++){{hi_p=MathMax(hi_p,high[i+k]);lo_p=MathMin(lo_p,low[i+k]);}}
   return -100*(hi_p-close[i])/(hi_p-lo_p+1e-9)/100.0+0.5;"""

def mql_donchian_pos(period):
    return f"""
   double dh=high[i],dl=low[i];
   for(int k=1;k<{period}&&i+k<needed;k++){{dh=MathMax(dh,high[i+k]);dl=MathMin(dl,low[i+k]);}}
   return (close[i]-dl)/(dh-dl+1e-9);"""

def mql_donchian_width(period):
    return f"""
   double dh=high[i],dl=low[i];
   for(int k=1;k<{period}&&i+k<needed;k++){{dh=MathMax(dh,high[i+k]);dl=MathMin(dl,low[i+k]);}}
   return (dh-dl)/(close[i]+1e-9);"""

def mql_diff_sma(period):
    return f"""
   double s=0; int c2=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{s+=close[i+k];c2++;}}
   double m=(c2>0)?s/c2:close[i];
   return (close[i]-m)/(close[i]+1e-9);"""

def mql_diff_ema(period):
    return f"""
   double alpha=2.0/({period}.0+1.0);
   double em=close[needed-1];
   for(int k=needed-2;k>=i;k--) em=alpha*close[k]+(1-alpha)*em;
   return (close[i]-em)/(close[i]+1e-9);"""

def mql_kc_pos():
    return """
   // KC pos (ema20 ± 1.5*atr14)
   double alpha_ema=2.0/21.0, alpha_rma=1.0/14.0;
   double em=close[needed-1];
   double rm=high[needed-1]-low[needed-1];
   for(int k=needed-2;k>=i;k--){
      em=alpha_ema*close[k]+(1-alpha_ema)*em;
      double hl=high[k]-low[k],hpc=MathAbs(high[k]-close[k+1]),lpc=MathAbs(low[k]-close[k+1]);
      rm=alpha_rma*MathMax(hl,MathMax(hpc,lpc))+(1-alpha_rma)*rm;
   }
   double kc_u=em+1.5*rm, kc_d=em-1.5*rm;
   return (close[i]-kc_d)/(kc_u-kc_d+1e-9);"""

def mql_kc_width():
    return """
   double alpha_ema=2.0/21.0, alpha_rma=1.0/14.0;
   double em=close[needed-1];
   double rm=high[needed-1]-low[needed-1];
   for(int k=needed-2;k>=i;k--){
      em=alpha_ema*close[k]+(1-alpha_ema)*em;
      double hl=high[k]-low[k],hpc=MathAbs(high[k]-close[k+1]),lpc=MathAbs(low[k]-close[k+1]);
      rm=alpha_rma*MathMax(hl,MathMax(hpc,lpc))+(1-alpha_rma)*rm;
   }
   double kc_u=em+1.5*rm, kc_d=em-1.5*rm;
   return (kc_u-kc_d)/(close[i]+1e-9);"""

def mql_ichi(part):
    if part == 'tenkan':
        return """
   double hi=high[i],lo=low[i];
   for(int k=1;k<9&&i+k<needed;k++){hi=MathMax(hi,high[i+k]);lo=MathMin(lo,low[i+k]);}
   return (hi+lo)/2.0;"""
    elif part == 'kijun':
        return """
   double hi=high[i],lo=low[i];
   for(int k=1;k<26&&i+k<needed;k++){hi=MathMax(hi,high[i+k]);lo=MathMin(lo,low[i+k]);}
   return (hi+lo)/2.0;"""
    elif part == 'senkou_a':
        return """
   double hi9=high[i],lo9=low[i]; for(int k=1;k<9&&i+k<needed;k++){hi9=MathMax(hi9,high[i+k]);lo9=MathMin(lo9,low[i+k]);}
   double tk=(hi9+lo9)/2.0;
   double hi26=high[i],lo26=low[i]; for(int k=1;k<26&&i+k<needed;k++){hi26=MathMax(hi26,high[i+k]);lo26=MathMin(lo26,low[i+k]);}
   double kj=(hi26+lo26)/2.0;
   return (tk+kj)/2.0;"""
    elif part == 'senkou_b':
        return """
   double hi=high[i],lo=low[i];
   for(int k=1;k<52&&i+k<needed;k++){hi=MathMax(hi,high[i+k]);lo=MathMin(lo,low[i+k]);}
   return (hi+lo)/2.0;"""
    elif part == 'cloud_pos':
        return """
   double hi9=high[i],lo9=low[i]; for(int k=1;k<9&&i+k<needed;k++){hi9=MathMax(hi9,high[i+k]);lo9=MathMin(lo9,low[i+k]);}
   double tk=(hi9+lo9)/2.0;
   double hi26=high[i],lo26=low[i]; for(int k=1;k<26&&i+k<needed;k++){hi26=MathMax(hi26,high[i+k]);lo26=MathMin(lo26,low[i+k]);}
   double kj=(hi26+lo26)/2.0;
   double sa=(tk+kj)/2.0;
   double hi52=high[i],lo52=low[i]; for(int k=1;k<52&&i+k<needed;k++){hi52=MathMax(hi52,high[i+k]);lo52=MathMin(lo52,low[i+k]);}
   double sb=(hi52+lo52)/2.0;
   double top=MathMax(sa,sb), bot=MathMin(sa,sb);
   return (close[i]-bot)/(top-bot+1e-9);"""
    elif part == 'cloud_width':
        return """
   double hi9=high[i],lo9=low[i]; for(int k=1;k<9&&i+k<needed;k++){hi9=MathMax(hi9,high[i+k]);lo9=MathMin(lo9,low[i+k]);}
   double tk=(hi9+lo9)/2.0;
   double hi26=high[i],lo26=low[i]; for(int k=1;k<26&&i+k<needed;k++){hi26=MathMax(hi26,high[i+k]);lo26=MathMin(lo26,low[i+k]);}
   double kj=(hi26+lo26)/2.0;
   double sa=(tk+kj)/2.0;
   double hi52=high[i],lo52=low[i]; for(int k=1;k<52&&i+k<needed;k++){hi52=MathMax(hi52,high[i+k]);lo52=MathMin(lo52,low[i+k]);}
   double sb=(hi52+lo52)/2.0;
   double top=MathMax(sa,sb), bot=MathMin(sa,sb);
   return (top-bot)/(close[i]+1e-9);"""
    else:  # tk_cross
        return """
   double hi9=high[i],lo9=low[i]; for(int k=1;k<9&&i+k<needed;k++){hi9=MathMax(hi9,high[i+k]);lo9=MathMin(lo9,low[i+k]);}
   double tk=(hi9+lo9)/2.0;
   double hi26=high[i],lo26=low[i]; for(int k=1;k<26&&i+k<needed;k++){hi26=MathMax(hi26,high[i+k]);lo26=MathMin(lo26,low[i+k]);}
   double kj=(hi26+lo26)/2.0;
   return tk-kj;"""

def mql_roll(period, stat):
    if stat == 'mean':
        return f"""
   double s=0; int c2=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{s+=close[i+k];c2++;}}
   return (c2>0)?s/c2:close[i];"""
    elif stat == 'std':
        return f"""
   double s=0; int c2=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{s+=close[i+k];c2++;}}
   double m=(c2>0)?s/c2:close[i];
   double sq=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{double d=close[i+k]-m;sq+=d*d;}}
   return MathSqrt((c2>0)?sq/c2:0);"""
    elif stat == 'skew':
        return f"""
   double s=0; int c2=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{s+=close[i+k];c2++;}}
   double m=(c2>0)?s/c2:close[i];
   double sq=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{double d=close[i+k]-m;sq+=d*d;}}
   double std=MathSqrt((c2>0)?sq/c2:0);
   if(std<1e-12) return 0.0;
   double skew=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{double d=(close[i+k]-m)/std;skew+=d*d*d;}}
   return (c2>0)?skew/c2:0;"""
    elif stat == 'kurt':
        return f"""
   double s=0; int c2=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{s+=close[i+k];c2++;}}
   double m=(c2>0)?s/c2:close[i];
   double sq=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{double d=close[i+k]-m;sq+=d*d;}}
   double std=MathSqrt((c2>0)?sq/c2:0);
   if(std<1e-12) return 0.0;
   double kurt=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{double d=(close[i+k]-m)/std;kurt+=d*d*d*d;}}
   return (c2>0)?kurt/c2-3.0:0;"""
    else:  # zscore
        return f"""
   double s=0; int c2=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{s+=close[i+k];c2++;}}
   double m=(c2>0)?s/c2:close[i];
   double sq=0;
   for(int k=0;k<{period}&&i+k<needed;k++){{double d=close[i+k]-m;sq+=d*d;}}
   double std=MathSqrt((c2>0)?sq/c2:0);
   return (std>1e-12)?(close[i]-m)/std:0.0;"""

def mql_candle(part):
    if part == 'body':
        return "   return close[i]-open[i];"
    elif part == 'upper_w':
        return "   return high[i]-MathMax(close[i],open[i]);"
    elif part == 'lower_w':
        return "   return MathMin(close[i],open[i])-low[i];"
    elif part == 'tr':
        return """
   double hl=high[i]-low[i];
   double hpc=(i+1<needed)?MathAbs(high[i]-close[i+1]):hl;
   double lpc=(i+1<needed)?MathAbs(low[i]-close[i+1]):hl;
   return MathMax(hl,MathMax(hpc,lpc));"""
    elif part == 'is_doji':
        return """
   double hl=high[i]-low[i]+1e-9;
   return (MathAbs(close[i]-open[i])<hl*0.1)?1.0:0.0;"""
    elif part == 'is_bull_engulf':
        if_i1 = "i+1<needed && "
        return f"""
   if(!(i+1<needed)) return 0.0;
   double body_cur=close[i]-open[i];
   double body_prv=close[i+1]-open[i+1];
   return (body_cur>0&&body_prv<0&&open[i]<=close[i+1]&&close[i]>=open[i+1])?1.0:0.0;"""
    elif part == 'is_bear_engulf':
        return """
   if(!(i+1<needed)) return 0.0;
   double body_cur=close[i]-open[i];
   double body_prv=close[i+1]-open[i+1];
   return (body_cur<0&&body_prv>0&&open[i]>=close[i+1]&&close[i]<=open[i+1])?1.0:0.0;"""
    elif part == 'is_hammer':
        return """
   double body=MathAbs(close[i]-open[i]);
   double lower=MathMin(close[i],open[i])-low[i];
   double upper=high[i]-MathMax(close[i],open[i]);
   return (lower>body*2&&upper<body*0.5)?1.0:0.0;"""
    elif part == 'is_inv_hammer':
        return """
   double body=MathAbs(close[i]-open[i]);
   double lower=MathMin(close[i],open[i])-low[i];
   double upper=high[i]-MathMax(close[i],open[i]);
   return (upper>body*2&&lower<body*0.5)?1.0:0.0;"""

def mql_bos(part):
    if part == 'hh':
        return "   return (i+1<needed&&high[i]>high[i+1])?1.0:0.0;"
    elif part == 'hl':
        return "   return (i+1<needed&&low[i]>low[i+1])?1.0:0.0;"
    elif part == 'lh':
        return "   return (i+1<needed&&high[i]<high[i+1])?1.0:0.0;"
    elif part == 'll':
        return "   return (i+1<needed&&low[i]<low[i+1])?1.0:0.0;"
    elif part == 'bos':
        return """
   if(i+1>=needed) return 0.0;
   bool hl=(low[i]>low[i+1]), hh=(high[i]>high[i+1]);
   bool lh=(high[i]<high[i+1]), ll=(low[i]<low[i+1]);
   bool bull_bos=(i+2<needed&&low[i+1]>low[i+2]&&hh);
   bool bear_bos=(i+2<needed&&high[i+1]<high[i+2]&&ll);
   return (bull_bos||bear_bos)?1.0:0.0;"""
    elif part == 'bos_dir':
        return """
   if(i+1>=needed) return 0.0;
   bool hh=(high[i]>high[i+1]), ll=(low[i]<low[i+1]);
   bool bull_bos=(i+2<needed&&low[i+1]>low[i+2]&&hh);
   bool bear_bos=(i+2<needed&&high[i+1]<high[i+2]&&ll);
   return bull_bos?1.0:(bear_bos?-1.0:0.0);"""
    else:  # bos_bars, bos_time, etc -> 0
        return "   return 0.0;"

def mql_session(part):
    return f"""
   MqlDateTime mdt; TimeToStruct(GetBarTime(i),mdt);
   int hr=mdt.hour;
   {"int dow=mdt.day_of_week;" if 'dow' in part else ""}
   {"return MathSin(2.0*M_PI*hr/24.0);" if part=='hour_sin' else ""}
   {"return MathCos(2.0*M_PI*hr/24.0);" if part=='hour_cos' else ""}
   {"return MathSin(2.0*M_PI*dow/5.0);" if part=='dow_sin' else ""}
   {"return MathCos(2.0*M_PI*dow/5.0);" if part=='dow_cos' else ""}
   {"return (hr>=0&&hr<9)?1.0:0.0;" if part=='is_tokyo' else ""}
   {"return (hr>=7&&hr<16)?1.0:0.0;" if part=='is_london' else ""}
   {"return (hr>=13&&hr<22)?1.0:0.0;" if part=='is_ny' else ""}
   {"return (hr>=13&&hr<16)?1.0:0.0;" if part=='is_overlap' else ""}"""

def mql_ret(p):
    return f"""
   double ref=(i+{p}<needed)?close[i+{p}]:close[needed-1];
   return (close[i]-ref)/(ref+1e-9);"""

def mql_vol(p):
    return f"""
   double s=0; int c2=0;
   for(int k=0;k<{p}&&i+k<needed;k++){{s+=vol[i+k];c2++;}}
   return (c2>0)?s/c2:0;"""

def mql_momentum(p):
    return f"""
   return (i+{p}<needed)?close[i]-close[i+{p}]:0.0;"""

def mql_volatility(p):
    return f"""
   double s=0;int c2=0;
   for(int k=0;k<{p}&&i+k<needed;k++){{s+=close[i+k];c2++;}}
   double m=(c2>0)?s/c2:close[i];
   double sq=0;
   for(int k=0;k<{p}&&i+k<needed;k++){{double d=close[i+k]-m;sq+=d*d;}}
   return MathSqrt((c2>0)?sq/c2:0);"""

def mql_price_diff(p):
    return f"""
   return (i+{p}<needed)?close[i]-close[i+{p}]:0.0;"""

def mql_psar():
    return "   return close[i]; // PSAR simplified"

def mql_pivot(part):
    if part == 'pivot':
        return """
   if(i+1>=needed) return close[i];
   return (high[i+1]+low[i+1]+close[i+1])/3.0;"""
    elif part == 'r1':
        return """
   if(i+1>=needed) return close[i];
   double pp=(high[i+1]+low[i+1]+close[i+1])/3.0;
   return 2*pp-low[i+1];"""
    elif part == 'r2':
        return """
   if(i+1>=needed) return close[i];
   double pp=(high[i+1]+low[i+1]+close[i+1])/3.0;
   return pp+(high[i+1]-low[i+1]);"""
    elif part == 's1':
        return """
   if(i+1>=needed) return close[i];
   double pp=(high[i+1]+low[i+1]+close[i+1])/3.0;
   return 2*pp-high[i+1];"""
    elif part == 's2':
        return """
   if(i+1>=needed) return close[i];
   double pp=(high[i+1]+low[i+1]+close[i+1])/3.0;
   return pp-(high[i+1]-low[i+1]);"""

# ──────────────────────────────────────────────────────────────────────────────
# 特徴量名 → MQL5 コード生成
# ──────────────────────────────────────────────────────────────────────────────
def get_code(name: str) -> str:
    """特徴量名から MQL5 計算コードの本体を返す"""
    if name.startswith('extra_random_feat_'):
        return "   return 0.0; // dummy feature"

    # SMA
    m = re.match(r'^sma_(\d+)$', name)
    if m: return mql_sma(int(m[1]))

    # EMA
    m = re.match(r'^ema_(\d+)$', name)
    if m: return mql_ema(int(m[1]))

    # WMA
    m = re.match(r'^wma_(\d+)$', name)
    if m: return mql_wma(int(m[1]))

    # HMA
    m = re.match(r'^hma_(\d+)$', name)
    if m: return mql_hma(int(m[1]))

    # MACD
    m = re.match(r'^macd_(\d+)_(\d+)_(\d+)$', name)
    if m: return mql_macd(int(m[1]),int(m[2]),int(m[3]),'line')
    m = re.match(r'^macdsig_(\d+)_(\d+)_(\d+)$', name)
    if m: return mql_macd(int(m[1]),int(m[2]),int(m[3]),'sig')
    m = re.match(r'^macdhist_(\d+)_(\d+)_(\d+)$', name)
    if m: return mql_macd(int(m[1]),int(m[2]),int(m[3]),'hist')

    # RSI
    m = re.match(r'^rsi_(\d+)$', name)
    if m: return mql_rsi(int(m[1]))

    # Stoch K
    m = re.match(r'^stoch_k_(\d+)_(\d+)$', name)
    if m: return mql_stoch(int(m[1]),int(m[2]),'k')

    # Stoch D
    m = re.match(r'^stoch_d_(\d+)_(\d+)$', name)
    if m: return mql_stoch(int(m[1]),int(m[2]),'d')

    # BB
    m = re.match(r'^bb_pos_(\d+)$', name)
    if m: return mql_bb(int(m[1]),'pos')
    m = re.match(r'^bb_width_(\d+)$', name)
    if m: return mql_bb(int(m[1]),'width')

    # ATR
    m = re.match(r'^atr_(\d+)$', name)
    if m: return mql_atr(int(m[1]))

    # ADX, PDI, NDI
    m = re.match(r'^adx_(\d+)$', name)
    if m: return mql_adx(int(m[1]))
    m = re.match(r'^pdi_(\d+)$', name)
    if m: return mql_pdi(int(m[1]))
    m = re.match(r'^ndi_(\d+)$', name)
    if m: return mql_ndi(int(m[1]))

    # CCI
    m = re.match(r'^cci_(\d+)$', name)
    if m: return mql_cci(int(m[1]))

    # WR
    m = re.match(r'^wr_(\d+)$', name)
    if m: return mql_wr(int(m[1]))

    # PSAR
    if name == 'psar': return mql_psar()

    # Pivot
    if name == 'pivot': return mql_pivot('pivot')
    if name == 'r1': return mql_pivot('r1')
    if name == 'r2': return mql_pivot('r2')
    if name == 's1': return mql_pivot('s1')
    if name == 's2': return mql_pivot('s2')

    # Donchian
    m = re.match(r'^donchian_pos_(\d+)$', name)
    if m: return mql_donchian_pos(int(m[1]))
    m = re.match(r'^donchian_width_(\d+)$', name)
    if m: return mql_donchian_width(int(m[1]))

    # KC
    if name == 'kc_pos_20': return mql_kc_pos()
    if name == 'kc_width_20': return mql_kc_width()

    # Diff SMA/EMA
    m = re.match(r'^diff_sma_(\d+)$', name)
    if m: return mql_diff_sma(int(m[1]))
    m = re.match(r'^diff_ema_(\d+)$', name)
    if m: return mql_diff_ema(int(m[1]))

    # Ichimoku
    if name == 'ichi_tenkan': return mql_ichi('tenkan')
    if name == 'ichi_kijun': return mql_ichi('kijun')
    if name == 'ichi_senkou_a': return mql_ichi('senkou_a')
    if name == 'ichi_senkou_b': return mql_ichi('senkou_b')
    if name == 'ichi_cloud_pos': return mql_ichi('cloud_pos')
    if name == 'ichi_cloud_width': return mql_ichi('cloud_width')
    if name == 'ichi_tk_cross': return mql_ichi('tk_cross')

    # Rolling stats
    m = re.match(r'^roll_mean_(\d+)$', name)
    if m: return mql_roll(int(m[1]),'mean')
    m = re.match(r'^roll_std_(\d+)$', name)
    if m: return mql_roll(int(m[1]),'std')
    m = re.match(r'^roll_skew_(\d+)$', name)
    if m: return mql_roll(int(m[1]),'skew')
    m = re.match(r'^roll_kurt_(\d+)$', name)
    if m: return mql_roll(int(m[1]),'kurt')
    m = re.match(r'^zscore_(\d+)$', name)
    if m: return mql_roll(int(m[1]),'zscore')

    # Candle
    if name == 'body': return mql_candle('body')
    if name == 'upper_w': return mql_candle('upper_w')
    if name == 'lower_w': return mql_candle('lower_w')
    if name == 'tr': return mql_candle('tr')
    if name == 'is_doji': return mql_candle('is_doji')
    if name == 'is_bull_engulf': return mql_candle('is_bull_engulf')
    if name == 'is_bear_engulf': return mql_candle('is_bear_engulf')
    if name == 'is_hammer': return mql_candle('is_hammer')
    if name == 'is_inv_hammer': return mql_candle('is_inv_hammer')

    # BOS
    if name == 'hh': return mql_bos('hh')
    if name == 'hl': return mql_bos('hl')
    if name == 'lh': return mql_bos('lh')
    if name == 'll': return mql_bos('ll')
    if name == 'bos': return mql_bos('bos')
    if name == 'bos_dir': return mql_bos('bos_dir')
    if name in ('bos_bars','bos_time','bos_pips','bos_fibo_zone',
                'bos_retest','bos_move_pips','bos_ema_sync',
                'bos_macd_sync','bos_rsi_sync'): return mql_bos('other')

    # Session
    if name == 'hour_sin': return mql_session('hour_sin')
    if name == 'hour_cos': return mql_session('hour_cos')
    if name == 'dow_sin': return mql_session('dow_sin')
    if name == 'dow_cos': return mql_session('dow_cos')
    if name == 'is_tokyo': return mql_session('is_tokyo')
    if name == 'is_london': return mql_session('is_london')
    if name == 'is_ny': return mql_session('is_ny')
    if name == 'is_overlap': return mql_session('is_overlap')

    # ret1_p, ret5_p, vol_p  (p = lag)
    m = re.match(r'^ret1_(\d+)$', name)
    if m: return mql_ret(int(m[1]))
    m = re.match(r'^ret5_(\d+)$', name)
    if m: return mql_ret(5)   # ret5 = pct_change(5)
    m = re.match(r'^vol_(\d+)$', name)
    if m: return mql_vol(int(m[1]))

    # momentum_p, volatility_p, price_diff_p
    m = re.match(r'^momentum_(\d+)$', name)
    if m: return mql_momentum(int(m[1]))
    m = re.match(r'^volatility_(\d+)$', name)
    if m: return mql_volatility(int(m[1]))
    m = re.match(r'^price_diff_(\d+)$', name)
    if m: return mql_price_diff(int(m[1]))

    # fallback
    return f"   return 0.0; // unknown: {name}"

# ──────────────────────────────────────────────────────────────────────────────
# ファイル生成
# ──────────────────────────────────────────────────────────────────────────────
print(f"Generating {N_GROUPS} feature files in {OUT_DIR} ...")

for idx, name in enumerate(BASE_COLS):
    fname = OUT_DIR / f"feat_{idx:03d}.mqh"
    func_name = f"CalcFeat_{idx:03d}"
    code = get_code(name)

    needs_time = 'GetBarTime' in code
    header = f"""// feat_{idx:03d}.mqh  — {name}
// Auto-generated by gen_feat_mqh.py  (feature index {idx})
// Python: features.py BASE_FEATURE_COLS[{idx}] = "{name}"
#ifndef FEAT_{idx:03d}_MQH
#define FEAT_{idx:03d}_MQH

{"// Requires GetBarTime(int i) defined in main EA" if needs_time else ""}

double {func_name}(
   const double &close[], const double &high[],
   const double &low[],  const double &open[],
   const double &vol[],  const int needed, const int i)
{{
{code}
}}

#endif // FEAT_{idx:03d}_MQH
"""
    fname.write_text(header, encoding='utf-8')
    if (idx+1) % 100 == 0:
        print(f"  {idx+1}/{N_GROUPS} done")

print("All done!")
