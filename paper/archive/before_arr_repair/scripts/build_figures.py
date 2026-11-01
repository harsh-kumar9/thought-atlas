"""Build compact manuscript figures from frozen input tables. No model fitting.

Run from repo root: MPLCONFIGDIR=/private/tmp/thought_atlas_mpl .venv/bin/python paper/scripts/build_figures.py
"""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

P=Path(__file__).resolve().parents[1]; ROOT=P.parent
S=ROOT/'docs/paper_synthesis_2026-10-31'; D=P/'data/figure_inputs'; F=P/'figures'
D.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
origins={
 'amount.csv':S/'sources/ada_corrected_atlas/02_amount_shape/behavior_amount_by_cell.csv',
 'gains.csv':S/'sources/ada_corrected_atlas/09_signatures/domain_signature_models.csv',
 'ladder.csv':P/'data/signature_ladder.csv',
 'random_timing.csv':P/'data/random_timing_control/summary.csv',
 'trajectory.csv':S/'reframing/tables/trajectory_model_means.csv',
 'distances.csv':S/'reframing/tables/domain_distances.csv',
 'noise.csv':S/'sources/ada_mode_control/data/f1/analysis/noise_floor.csv',
 'shifts.csv':S/'reframing/sources/mode_control/data/transplant/shifts.csv',
 'vector_cosines.csv':S/'reframing/sources/mode_control/data/transplant/vector_cosines.csv',
 'shift_cosines.csv':S/'reframing/sources/mode_control/data/transplant/shift_cosines.csv',
 'readiness.csv':S/'reframing/sources/mode_control/data/readiness/math/paired.csv',
 'pairwise_gemma.csv':S/'sources/ada_mode_control/data/pairwise/summary__google_gemma-4-31B-it.csv',
 'pairwise_qwen.csv':S/'sources/ada_mode_control/data/pairwise/summary__Qwen_Qwen3.5-27B.csv',
 'examples.parquet':S/'reframing/sources/example_segments.parquet',
 'example_selection.csv':S/'reframing/tables/example_selection.csv',
}
for name,src in origins.items():shutil.copy2(src,D/name)
# A restrained, print-oriented design. Color is always paired with position,
# signed values, marker shape, or line pattern.
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex
from matplotlib.lines import Line2D
INK='#30383C'; MUTED='#626C70'; RULE='#D7DDDE'; PALE='#F4F6F6'
BLUE='#5588A4'; OCHRE='#AC834C'; TEAL='#548B7F'; PLUM='#927C9F'
GRAY='#7A858C'
SEQ=LinearSegmentedColormap.from_list('atlas_blue',['#FAFBFB','#E2ECF0','#B8CED9','#7EA7BD'])
DIV=LinearSegmentedColormap.from_list('atlas_ochre_blue',['#D4B68B','#F8F8F5','#9ABED0'])
plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':8,'text.color':INK,
 'axes.labelcolor':INK,'axes.edgecolor':RULE,'axes.linewidth':.5,
 'axes.titlesize':10,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
 'xtick.color':MUTED,'ytick.color':INK,'xtick.major.width':.5,'ytick.major.width':.5,
 'xtick.major.size':2.5,'ytick.major.size':0,'legend.fontsize':7.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
 'axes.spines.bottom':False,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.titleweight':'normal','savefig.facecolor':'white',
 'lines.solid_capstyle':'round','lines.dash_capstyle':'round',
 'axes.axisbelow':True})
B=['Question_and_Answering','Perspective_Shift','Conflict_of_Perspectives','Reconciliation','verification','backtracking','subgoal','backward_chaining']
L=['Questioning','Perspective shift','Conflict','Reconciliation','Verification','Backtracking','Subgoals','Backward chaining']
C=[BLUE,PLUM,'#BC8B98',OCHRE,TEAL,'#BA8965','#8C985F',GRAY]
DOM=['math','code','gpqa','planning','moral','idea','safety','security']
MODELS=['reasoner','gemma4_e4b','gemma4_12b','gemma4_31b','qwen35_4b','qwen35_9b','qwen35_27b']
MODEL_NAMES=json.loads((P/'model_names.json').read_text())
def model_label(key):return MODEL_NAMES[key]['display'].replace('DeepSeek-R1-Distill-','DeepSeek-R1-\nDistill-')
ML=[model_label(m) for m in MODELS]
def domain(d):return 'GPQA' if d=='gpqa' else d.title()
def tint(c,amount=.56):return to_hex(np.array(to_rgb(c))*(1-amount)+amount)
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(F/f'{name}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.07)
 plt.close(fig)
def title(ax,text):
 ax.set_title(text,loc='left',pad=10,fontfamily='STIXGeneral',fontsize=10.5,color=INK)
def zero(ax,orientation='x'):
 getattr(ax,'axvline' if orientation=='x' else 'axhline')(0,color='#9CA6A9',lw=.7,zorder=1)
def guides(ax,axis='x'):
 ax.grid(axis=axis,color=RULE,lw=.45,alpha=.55)
def colorbar(fig,im,ax=None,**kwargs):
 cb=fig.colorbar(im,ax=ax,**kwargs);cb.outline.set_visible(False)
 cb.ax.tick_params(length=2,width=.45,labelsize=7)
 return cb
def heat(ax,a,rows,cols,cmap=SEQ,vmin=0,vmax=30,fmt='.1f',fontsize=8):
 im=ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
 ax.set_yticks(range(len(rows)),rows);ax.set_xticks(range(len(cols)),cols)
 ax.tick_params(axis='both',length=0,pad=6)
 # Fine white gutters separate cells without drawing a heavy grid.
 ax.set_xticks(np.arange(-.5,len(cols),1),minor=True)
 ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
 ax.grid(which='minor',color='white',linewidth=1.2)
 ax.tick_params(which='minor',bottom=False,left=False)
 for i in range(len(rows)):
  for j in range(len(cols)):
   if np.isfinite(a[i,j]):
    ax.text(j,i,format(a[i,j],fmt),ha='center',va='center',fontsize=fontsize,color=INK)
 return im
def interval(ax,v,y,lo,hi,color,marker='o',label=None,hollow=False):
 ax.errorbar(v,y,xerr=[np.asarray(v)-np.asarray(lo),np.asarray(hi)-np.asarray(v)],
  fmt=marker,ms=4.2,elinewidth=1.1,capsize=1.5,capthick=.8,
  color=color,mfc='white' if hollow else tint(color,.25),mec=color,mew=.8,label=label,zorder=3)

def atlas():
 a=pd.read_csv(D/'amount.csv').groupby(['task_type','behavior']).mean_rate.mean().unstack().loc[DOM,B]*100
 fig=plt.figure(figsize=(7.1,4.35))
 g=fig.add_gridspec(2,1,height_ratios=[.85,1.15],hspace=.65)
 fig.subplots_adjust(left=.205,right=.95,top=.86,bottom=.16)
 ax=fig.add_subplot(g[0])
 cols=['Questioning','Perspective\nshift','Conflict','Reconcil-\niation','Verification','Backtracking','Subgoals','Backward\nchaining']
 im=heat(ax,a.to_numpy(),[domain(d) for d in DOM],cols,fontsize=7.8)
 ax.tick_params(axis='x',labelsize=6.9)
 ax.set_title('A   Domain profiles',loc='left',pad=33,fontfamily='STIXGeneral',fontsize=10.5,color=INK)
 # A shared numeric scale, but two explicitly separated annotation families.
 ax.axvline(3.5,color='white',lw=4,zorder=4)
 for lo,hi,col,mark,label in [(0,.5,BLUE,'o','Conversational'),(.5,1,OCHRE,'s','Cognitive operations')]:
  ax.plot([lo+.015,hi-.015],[1.035,1.035],transform=ax.transAxes,color=col,lw=1.2,clip_on=False)
  ax.text((lo+hi)/2,1.10,label,transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color=col)
  ax.scatter([lo+.045],[1.145],transform=ax.transAxes,marker=mark,s=17,facecolors='white',edgecolors=col,lw=.8,clip_on=False)
 colorbar(fig,im,ax=ax,pad=.018,fraction=.022,ticks=[0,10,20,30],label='Labeled segments (%)')
 ladder=pd.read_csv(D/'ladder.csv').set_index(['gen_model','signature_family','stage'])
 ax=fig.add_subplot(g[1])
 for i,m in enumerate(MODELS):
  length=ladder.loc[(m,'conversational','length')]
  ax.hlines(i,0,length.gain_bits,color=RULE,lw=1.3,zorder=1)
  interval(ax,[length.gain_bits],[i],[length.ci_low],[length.ci_high],GRAY,'D',hollow=True)
  for fam,col,off,mark in [('conversational',BLUE,-.18,'o'),('cognitive',OCHRE,.18,'s')]:
   amount=ladder.loc[(m,fam,'amount')]; full=ladder.loc[(m,fam,'full')]
   ax.plot([length.gain_bits,amount.gain_bits],[i,i+off],color=tint(col,.55),lw=1.5,zorder=1)
   ax.plot([amount.gain_bits,full.gain_bits],[i+off,i+off],color=col,lw=2.0,zorder=2)
   interval(ax,[amount.gain_bits],[i+off],[amount.ci_low],[amount.ci_high],col,mark,hollow=True)
   interval(ax,[full.gain_bits],[i+off],[full.ci_low],[full.ci_high],col,mark)
 ax.set_yticks(range(7),ML);ax.set_ylim(6.55,-.55);ax.set_xlim(-.025,2.06)
 ax.set_xticks([0,.5,1,1.5,2]);zero(ax);guides(ax)
 ax.set_xlabel('Held-out gain over uniform guessing (bits / trace)',fontsize=8)
 title(ax,'B   Predicting domain: length → amount → timing')
 handles=[Line2D([],[],marker='D',color=GRAY,ls='none',mfc='white',ms=4,label='Length only'),
          Line2D([],[],marker='o',color=INK,ls='none',mfc='white',ms=4,label='+ Amount'),
          Line2D([],[],marker='o',color=INK,ls='none',mfc=tint(INK,.25),ms=4,label='+ Timing')]
 fig.legend(handles=handles,frameon=False,loc='lower center',bbox_to_anchor=(.59,.005),ncol=3,handlelength=1,columnspacing=1.6)
 save(fig,'fig01_signatures')

def random_control():
 d=pd.read_csv(D/'random_timing.csv').set_index(['gen_model','signature_family'])
 fig,axs=plt.subplots(1,2,figsize=(7.1,2.85),sharey=True,gridspec_kw={'wspace':.16})
 fig.subplots_adjust(left=.225,right=.98,top=.86,bottom=.24)
 for ax,fam,col,mark,label in zip(axs,['conversational','cognitive'],[BLUE,OCHRE],['o','s'],['A   Conversational','B   Cognitive operations']):
  a=d.xs(fam,level='signature_family').loc[MODELS]
  for i,(_,r) in enumerate(a.iterrows()):
   ax.plot([r.random_gain_bits,r.timing_gain_bits],[i+.12,i-.12],color=tint(col,.68),lw=1,zorder=1)
  interval(ax,a.timing_gain_bits,np.arange(7)-.12,a.timing_ci_low,a.timing_ci_high,col,mark)
  interval(ax,a.random_gain_bits,np.arange(7)+.12,a.random_mean_ci_low,a.random_mean_ci_high,GRAY,'x')
  ax.set_yticks(range(7),ML);ax.set_ylim(6.5,-.5);ax.set_xlim(-.04,.215)
  ax.set_xticks([-.025,0,.05,.10,.15,.20]);ax.set_xticklabels(['−.025','0','.05','.10','.15','.20'])
  zero(ax);guides(ax);title(ax,label)
  ax.tick_params(labelsize=7.5)
 from matplotlib.legend_handler import HandlerTuple
 observed=(Line2D([],[],marker='o',ls='none',color=BLUE,mfc=tint(BLUE,.25),ms=4),
           Line2D([],[],marker='s',ls='none',color=OCHRE,mfc=tint(OCHRE,.25),ms=4))
 shuffled=Line2D([],[],marker='x',ls='none',color=GRAY,ms=4)
 fig.supxlabel('Added information beyond length + amount (bits / trace)',x=.60,y=.105,fontsize=8)
 fig.legend([observed,shuffled],['Observed timing','Shuffled timing (100 draws)'],handler_map={tuple:HandlerTuple(ndivide=None)},frameon=False,loc='lower center',bbox_to_anchor=(.60,-.025),ncol=2,fontsize=8,handlelength=2,columnspacing=1.4)
 save(fig,'figA3_random_timing_control')


def trajectories():
 x=pd.read_csv(D/'trajectory.csv').set_index(['gen_model','task_type','bin']);avg=x.groupby(['task_type','bin'])[B].mean()
 ds=['math','planning','moral','idea'];bs=['subgoal','verification','Perspective_Shift','backtracking']
 fig,axes=plt.subplots(4,4,figsize=(7.1,3.75),sharex=True,sharey='row',gridspec_kw={'wspace':.15,'hspace':.22})
 fig.subplots_adjust(left=.19,right=.98,top=.91,bottom=.13)
 for i,b in enumerate(bs):
  limit=np.ceil(x.reset_index().query('task_type in @ds')[b].max()/10)*10;col=C[B.index(b)]
  for j,d in enumerate(ds):
   ax=axes[i,j]
   for m in MODELS:ax.plot(np.arange(5,100,10),x.loc[(m,d),b].reindex(range(10)),color=tint(col,.62),lw=.6,zorder=1)
   ax.plot(np.arange(5,100,10),avg.loc[d,b].reindex(range(10)),color=col,lw=1.8,zorder=3)
   ax.set_xlim(0,100);ax.set_ylim(0,limit);ax.set_xticks([0,50,100]);ax.set_yticks([0,limit/2,limit]);guides(ax,'y')
   if i==0:ax.set_title(d.title(),fontfamily='STIXGeneral',fontsize=10.5)
   if j==0:
    row_label=['Subgoals\n(%)','Verification\n(%)','Perspective\nshift (%)','Backtracking\n(%)'][i]
    ax.set_ylabel(row_label,fontsize=8,rotation=0,ha='right',va='center',labelpad=13)
 fig.supxlabel('Relative position in reasoning trace (%)',x=.55,y=.025,fontsize=8)
 save(fig,'fig02_trajectories')

def demand():
 order=['qwen35_9b','gemma4_12b','reasoner','r1qwen7b','openthinker3_7b'];names=[model_label(m) for m in order]
 d=pd.read_csv(D/'noise.csv').set_index('model').loc[order]
 fig,axs=plt.subplots(1,2,figsize=(3.35,2.55),gridspec_kw={'wspace':.28})
 fig.subplots_adjust(left=.36,right=.98,top=.88,bottom=.23)
 for metric,col,off,lab,marker in [('demand_excess',BLUE,-.15,'Demand','o'),('format_excess',OCHRE,.15,'Format','s')]:
  interval(axs[0],d[metric],np.arange(5)+off,d[metric+'_lo'],d[metric+'_hi'],col,marker,lab)
 interval(axs[1],d.demand_minus_format,np.arange(5),d.dmf_lo,d.dmf_hi,TEAL,'D')
 for j,ax in enumerate(axs):
  ax.set_yticks(range(5),names if j==0 else ['']*5);ax.tick_params(labelsize=7)
  ax.set_ylim(4.5,-.5);zero(ax);guides(ax)
 axs[0].set_xticks([0,.4,.8]);axs[1].set_xticks([0,.4,.8])
 title(axs[0],'A  Excess');title(axs[1],'B  Contrast')
 axs[0].set_xlabel('Excess distance',fontsize=7.2)
 axs[1].set_xlabel('Demand − format',fontsize=7.2)
 fig.legend(*axs[0].get_legend_handles_labels(),frameon=False,loc='lower center',bbox_to_anchor=(.65,.005),ncol=2,fontsize=7,handlelength=1.3,columnspacing=.8)
 save(fig,'fig03_task_demand')


def transfer():
 x=pd.read_csv(P/'data/mode_control/data/transplant/outcomes.csv');arms=['sig_'+d for d in ['math','gpqa','planning','idea','moral']];targets=['math','gpqa','planning']
 fig,axs=plt.subplots(1,2,figsize=(3.35,2.45),gridspec_kw={'wspace':.12})
 fig.subplots_adjust(left=.19,right=.985,top=.85,bottom=.28)
 for j,(ax,budget) in enumerate(zip(axs,[12288,32768])):
  d=x[x.outcome.eq('correct')&x.budget.eq(budget)&x.arm.isin(arms)]
  a=d.pivot(index='arm',columns='target',values='delta').loc[arms,targets].to_numpy()*100
  im=heat(ax,a,['Math','GPQA','Planning','Idea','Moral'] if j==0 else ['']*5,['Math','GPQA','Planning'],cmap=DIV,vmin=-90,vmax=90,fmt='+.1f',fontsize=7.6)
  ax.tick_params(axis='both',labelsize=7,pad=3)
  for i in range(3):ax.add_patch(Rectangle((i-.48,i-.48),.96,.96,fill=False,edgecolor=INK,lw=.65))
  title(ax,('A' if j==0 else 'B')+f'  {budget:,} tokens')
 cax=fig.add_axes([.30,.10,.60,.035])
 cb=colorbar(fig,im,cax=cax,orientation='horizontal',ticks=[-90,0,90])
 cb.ax.tick_params(labelsize=6.5,pad=1)
 cb.set_label('Accuracy change (pp)',fontsize=7,labelpad=1)
 save(fig,'fig04_cross_domain_transfer')
 x[x.arm.isin(arms)].to_csv(P/'data/transfer_figure_cells.csv',index=False)


def rl():
 sh=pd.read_csv(D/'shifts.csv').query('target=="math"').set_index('arm')
 sc=pd.read_csv(D/'shift_cosines.csv').query('target=="math"').set_index('arm')
 vc=pd.read_csv(D/'vector_cosines.csv').set_index('vector')
 arms=['ol_math_A2','rl_zeroT_s60','sealdir_matched','ol_random'];names=['Atlas A2','RL from zero','SEAL, matched norm','Random']
 fig=plt.figure(figsize=(7.1,3.95));g=fig.add_gridspec(2,2,hspace=.55,wspace=.24)
 fig.subplots_adjust(left=.175,right=.98,top=.91,bottom=.16)
 ax=fig.add_subplot(g[0,0]);v=sh.loc[arms,['d_'+b for b in B]].to_numpy(float)
 heat(ax,v,names,['Q','P','C','R','V','B','S','K'],cmap=DIV,vmin=-8,vmax=8,fmt='+.1f',fontsize=8)
 ax.tick_params(axis='both',labelsize=8)
 title(ax,'A   Behavior change (pp)');ax.set_xlabel('Operation codes',fontsize=8)
 ax=fig.add_subplot(g[0,1]);ids=['rl_zero8k_T_s60','sealdir_L17_n6p75','ol_random'];y=np.arange(1,4)
 r=np.array([vc.loc[n,'ol_math_A2'] for n in ids]);b=sc.loc[arms[1:],'cos_with_A2_same_target'].to_numpy()
 ax.hlines(y,np.minimum(r,b),np.maximum(r,b),color=RULE,lw=1.4)
 ax.scatter(r,y,marker='o',s=25,facecolors='white',edgecolors=GRAY,linewidths=.9,label='Residual vector',zorder=3)
 ax.scatter(b,y,marker='D',s=22,facecolors=tint(BLUE,.2),edgecolors=BLUE,linewidths=.7,label='Behavior shift',zorder=3)
 ax.set_yticks(range(4),['']*4);ax.set_ylim(3.5,-.5);zero(ax);ax.set_xlim(-1,1.1);ax.set_xticks([-1,0,1])
 ax.set_xlabel('Cosine similarity to A2',fontsize=8)
 ax.legend(frameon=False,fontsize=7.2,loc='upper left',bbox_to_anchor=(0,1.05),ncol=2,handletextpad=.3,columnspacing=.7)
 title(ax,'B   Similarity to A2')
 d=pd.read_csv(P/'data/rl_budget_summary.csv');conditions=['baseline','ol_math_A2','rl_zeroT_s60','rl_a2initT_s60']
 labs=['Unsteered','Atlas A2','RL from zero','A2 + RL'];cols=[GRAY,BLUE,OCHRE,TEAL];markers=['o','s','^','D'];styles=['-', '--','-.',':']
 ax=fig.add_subplot(g[1,0])
 for cond,lab,col,marker,ls in zip(conditions,labs,cols,markers,styles):
  a=d[d.condition.eq(cond)].sort_values('budget')
  ax.plot(a.budget/1024,a.correct*100,color=col,marker=marker,ls=ls,ms=3.5,lw=1.4,mec=col,mfc=tint(col,.3),mew=.65,label=lab)
 ax.set_xticks([4,8,16,24,32]);ax.set_xlabel('Completion cutoff (×1,024)',fontsize=8)
 ax.set_ylabel('Correct answers (%)');ax.set_ylim(15,102);ax.set_yticks([20,40,60,80,100]);guides(ax,'y')
 title(ax,'C   Natural completion')
 ax2=fig.add_subplot(g[1,1]);d=pd.read_csv(D/'readiness.csv')
 for cond,col,off,marker in zip(conditions[1:],cols[1:],[-.18,0,.18],markers[1:]):
  a=d[d.condition.eq(cond)].set_index('cut').loc[[1024,2048,4096]];v=a.delta_vs_baseline.to_numpy()*100
  ax2.errorbar(np.arange(3)+off,v,yerr=[v-a.lo*100,a.hi*100-v],fmt=marker,color=col,ms=4,elinewidth=1.1,capsize=1.5,mfc=tint(col,.3),mec=col,mew=.7)
 zero(ax2,'y');ax2.set_xticks(range(3),['1,024','2,048','4,096']);ax2.set_xlabel('Forced-answer cutoff (tokens)',fontsize=8)
 ax2.set_ylabel('Accuracy change (pp)');ax2.set_yticks([-5,0,5,10]);guides(ax2,'y');title(ax2,'D   Early forced answering')
 fig.legend(*ax.get_legend_handles_labels(),frameon=False,loc='lower center',bbox_to_anchor=(.57,.005),ncol=4,handlelength=2.5,columnspacing=1.1,fontsize=8)
 save(fig,'fig05_steering_and_rl')

def ideas():
 specs=[('sched1','sus_math_L19','Checking vector'),('sched1','sus_idea_L19','Exploration vector'),('tx_idea','sig_math','Math-signature\nvector'),('tx_idea','prompt_math','Math-style prompt')]
 fig,axs=plt.subplots(1,2,figsize=(3.35,2.4),gridspec_kw={'wspace':.23})
 fig.subplots_adjust(left=.32,right=.98,top=.84,bottom=.28)
 for file,col,off,label,marker in [('pairwise_gemma.csv',BLUE,-.15,'Gemma4-31B-it','o'),('pairwise_qwen.csv',OCHRE,.15,'Qwen3.5-27B','s')]:
  d=pd.read_csv(D/file)
  for j,axis in enumerate(['originality','feasibility']):
   for i,(run,cond,_) in enumerate(specs):
    x=d[(d.run==run)&(d.condition==cond)&(d.axis==axis)].iloc[0]
    interval(axs[j],[x.net_win],[i+off],[x.lo],[x.hi],col,marker,label if i==0 else None)
 for j,ax in enumerate(axs):
  ax.set_yticks(range(4),[s[2] for s in specs] if j==0 else ['']*4);ax.set_ylim(3.5,-.5);zero(ax)
  ax.tick_params(labelsize=7);ax.set_xlim(-.8,.8);ax.set_xticks([-.5,0,.5]);guides(ax)
  ax.set_title('A  Originality' if j==0 else 'B  Feasibility',loc='left',fontfamily='STIXGeneral',fontsize=9,pad=8)
 fig.supxlabel('Net preference over baseline',x=.65,y=.14,fontsize=7.3)
 fig.legend(*axs[1].get_legend_handles_labels(),frameon=False,loc='lower center',bbox_to_anchor=(.52,-.015),ncol=2,fontsize=6.9,handlelength=1.2,columnspacing=.8,handletextpad=.3)
 save(fig,'fig06_open_task_tradeoffs')


def appendix():
 d=pd.read_csv(D/'distances.csv');a=np.zeros((8,8))
 for _,r in d.iterrows():i,j=DOM.index(r.domain_a),DOM.index(r.domain_b);a[i,j]=a[j,i]=r.distance_pp
 fig,ax=plt.subplots(figsize=(5.8,4.4));fig.subplots_adjust(left=.16,right=.88,top=.97,bottom=.16)
 im=heat(ax,a,[domain(d) for d in DOM],[domain(d) for d in DOM],vmax=12,fontsize=8.5)
 ax.tick_params(axis='x',rotation=45);plt.setp(ax.get_xticklabels(),ha='right',rotation_mode='anchor')
 colorbar(fig,im,ax=ax,fraction=.025,pad=.04,ticks=[0,4,8,12],label='RMS difference (pp)')
 save(fig,'figA1_domain_similarity')
 sel=pd.read_csv(D/'example_selection.csv').set_index('task_type');s=pd.read_parquet(D/'examples.parquet')
 fig,axs=plt.subplots(2,2,figsize=(7.1,3.4),gridspec_kw={'hspace':.55,'wspace':.14})
 fig.subplots_adjust(left=.045,right=.99,top=.90,bottom=.13)
 for ax,dom in zip(axs.flat,['math','planning','moral','idea']):
  r=sel.loc[dom];x=s[s.trace_id.eq(r.trace_id)].sort_values('seg_idx');n=int(x.n_segments.iloc[0]);a=np.zeros((8,n))
  for j,b in enumerate(B):a[j,x.seg_idx.to_numpy(int)]=x[b].to_numpy()*(j+1)
  ax.imshow(a,cmap=ListedColormap(['#F4F6F6']+[tint(c,.17) for c in C]),vmin=0,vmax=8,aspect='auto',interpolation='nearest',extent=[0,n,7.5,-.5])
  ax.set_yticks(range(8),['Q','P','C','R','V','B','S','K'],fontsize=7)
  ax.tick_params(axis='y',pad=4)
  ax.set_title(f'{dom.title()}  /  {n} segments, {int(r.n_new_tokens):,} tokens',loc='left',fontfamily='STIXGeneral',fontsize=9,pad=8)
 fig.supxlabel('Sentence index within each trace',x=.52,y=.02,fontsize=8)
 save(fig,'figA2_trace_examples')

import argparse
builders=dict(atlas=atlas,trajectories=trajectories,demand=demand,transfer=transfer,rl=rl,ideas=ideas,appendix=appendix,random_control=random_control)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only',nargs='+',choices=list(builders),help='Rebuild selected figures; default: all.')
args=parser.parse_args()
for name in args.only or builders:builders[name]()
manifest=dict(origins=[dict(file=str((D/k).relative_to(P)),source=str(v.relative_to(ROOT)),sha256=hashlib.sha256((D/k).read_bytes()).hexdigest()) for k,v in origins.items()],
 figures=[dict(file=str(p.relative_to(P)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(F.glob('*.pdf'))],
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(P/'figure_manifest.json').write_text(json.dumps(manifest,indent=2));print('Built manuscript figures: '+', '.join(args.only or builders))
