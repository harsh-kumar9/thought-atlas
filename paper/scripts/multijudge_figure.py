"""Compact three-family robustness figure; no judge is treated as ground truth."""
from build_figures import *
R=P/'revision/results/multijudge_analysis_indexed'
a=pd.read_csv(R/'pairwise_agreement.csv');a=a[a.source.eq('atlas')&a.domain.eq('all')]
c=pd.read_csv(R/'atlas_contrasts.csv');c=c[c.scope.eq('common_valid')]
fig,axs=plt.subplots(1,2,figsize=(7.1,3.5),gridspec_kw={'width_ratios':[1.05,1]})
fig.subplots_adjust(left=.15,right=.98,top=.82,bottom=.30,wspace=.80)
ax=axs[0]
for pair,offset,col,mark,label in [(('gemma','qwen'),-.20,BLUE,'o','Gemma / Qwen'),(('gemma','llama'),0,OCHRE,'s','Gemma / Llama'),(('qwen','llama'),.20,PLUM,'^','Qwen / Llama')]:
 d=a[a.judge_a.eq(pair[0])&a.judge_b.eq(pair[1])].set_index('label').loc[B]
 interval(ax,d.positive_agreement,np.arange(8)+offset,d.positive_agreement_lo,d.positive_agreement_hi,col,mark,label=label)
ax.set_yticks(range(8),L,fontsize=7);ax.set_ylim(7.6,-.6);ax.set_xlim(0,1);ax.set_xticks([0,.5,1]);ax.set_xlabel('Positive agreement');guides(ax)
ax.axhline(3.5,color=RULE,lw=.6)
title(ax,'A   Label agreement');ax.legend(loc='upper left',bbox_to_anchor=(-.25,-.25),frameon=False,fontsize=7)
ax=axs[1]
keys=[('idea','math','Perspective_Shift'),('idea','math','backtracking'),('planning','moral','verification'),('math','planning','subgoal')]
for judge,off,col,mark,label in [('gemma',-.20,BLUE,'o','Gemma4-31B-it'),('qwen',0,OCHRE,'s','Qwen3.5-27B'),('llama',.20,PLUM,'^','Llama-3.1-8B-Instruct')]:
 d=c[c.judge.eq(judge)].set_index(['high','low','label']).loc[keys]
 ax.plot(d.mean_difference*100,np.arange(4)+off,linestyle='none',color=col,marker=mark,ms=4,label=label)
 for y,(_,r) in enumerate(d.iterrows()):ax.text(22,y+off,f'{r.models_positive}/7',ha='left',va='center',fontsize=6.5,color=col)
ax.set_yticks(range(4),['Idea − math\nPerspective shift','Idea − math\nBacktracking','Planning − moral\nVerification','Math − planning\nSubgoals'],fontsize=7);ax.set_ylim(3.5,-.5);ax.set_xlim(-1,26);ax.set_xticks([0,10,20]);ax.set_xlabel('Prevalence difference (pp)');zero(ax);guides(ax)
title(ax,'B   Domain contrasts');ax.text(22,-.62,'Models +',fontsize=6.5,color=MUTED,ha='left')
ax.legend(loc='upper left',bbox_to_anchor=(-.18,-.25),frameon=False,fontsize=7)
save(fig,'figA10_multijudge')
for name in ['pairwise_agreement','parse_coverage','atlas_contrasts','paired_behavior_shifts','atlas_weighted_profiles']:
 shutil.copy2(R/f'{name}.csv',P/'supplement'/f'multijudge_{name}.csv')
