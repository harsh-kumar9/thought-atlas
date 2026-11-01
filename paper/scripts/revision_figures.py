"""Compact diagnostics added in the evidence-repair revision."""
from build_figures import *
R=P/'revision/results'

def fidelity():
 d=pd.read_csv(R/'behavior/fidelity.csv');d=d[~d.condition.str.contains('_x')]
 source=['math','gpqa','planning','idea','moral'];targets=source
 fig=plt.figure(figsize=(7.1,2.75));g=fig.add_gridspec(1,3,width_ratios=[1.3,1,1],wspace=.50)
 fig.subplots_adjust(left=.08,right=.99,top=.80,bottom=.26)
 ax=fig.add_subplot(g[0]);a=d.pivot(index='condition',columns='target',values='intended_cosine').loc[['sig_'+s for s in source],targets]
 heat(ax,a.to_numpy(),[domain(s) for s in source],[domain(s) for s in targets],cmap=DIV,vmin=-1,vmax=1,fmt='.2f',fontsize=7.7)
 ax.tick_params(axis='x',labelsize=7,rotation=35,pad=3);plt.setp(ax.get_xticklabels(),ha='right',rotation_mode='anchor')
 title(ax,'A   Intended–achieved cosine')
 dose=pd.read_csv(R/'behavior/dose_matched.csv');dose=dose[dose.target.eq('math')&dose.source.isin(['gpqa','idea'])]
 ax=fig.add_subplot(g[1]);ax2=fig.add_subplot(g[2])
 for source,col,marker in [('gpqa',BLUE,'o'),('idea',OCHRE,'s')]:
  a=dose[dose.source.eq(source)].sort_values('scale')
  ax.plot(a.scale,a.d_accuracy_32768*100,color=col,marker=marker,ms=4,label=domain(source))
  ax.plot(a.scale,a.d_answered_32768*100,color=col,ls=':',lw=1,marker=marker,mfc='white',ms=3)
  ax2.plot(a.scale,a.hard_loop_rate*100,color=col,marker=marker,ms=4)
 for a in [ax,ax2]:a.set_xticks([.5,1,1.5]);a.set_xlabel('Relative dose');guides(a,'y')
 title(ax,'B   Math outcome change');ax.set_ylim(-90,0);ax.set_yticks([-80,-40,0]);ax.set_ylabel('Percentage points')
 title(ax2,'C   Hard-loop prevalence');ax2.set_ylim(0,100);ax2.set_yticks([0,50,100]);ax2.set_ylabel('Attempts (%)')
 fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.62,.01),ncol=2,frameon=False)
 save(fig,'figA4_intervention_fidelity')

def similarity():
 d=pd.read_csv(R/'behavior/similarity_sensitivity.csv')
 fig,axs=plt.subplots(1,2,figsize=(7.1,2.5));fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.31,wspace=.30)
 for condition,col,marker,label in [('rl_zeroT_s60',OCHRE,'^','RL from zero'),('sealdir_matched',PLUM,'s','Matched SEAL'),('ol_random',GRAY,'x','Random')]:
  a=d[d.condition.eq(condition)&d.population.eq('all_paired')].sort_values('window')
  x=np.arange(4)
  axs[0].plot(x,a.cosine,color=col,marker=marker,ms=4,label=label);axs[0].fill_between(x,a.lo,a.hi,color=col,alpha=.10,lw=0)
  axs[1].plot(x,a.magnitude_ratio,color=col,marker=marker,ms=4);axs[1].fill_between(x,a.ratio_lo,a.ratio_hi,color=col,alpha=.10,lw=0)
 for ax in axs:ax.set_xticks(range(4),['1,024','2,048','4,096','8,192']);ax.set_xlabel('Maximum reasoning window (tokens)');guides(ax,'y')
 title(axs[0],'A   Direction relative to A2');axs[0].set_ylabel('Cosine');axs[0].set_ylim(-1,1);zero(axs[0],'y')
 title(axs[1],'B   Magnitude relative to A2');axs[1].set_ylabel('Norm ratio');axs[1].axhline(1,color=RULE,lw=.7)
 fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,.01),ncol=3,frameon=False)
 save(fig,'figA5_behavior_sensitivity')

def demand():
 d=pd.read_csv(R/'behavior/matched_demand_operations.csv').query('domain=="all"')
 models=['qwen35_9b','gemma4_12b','reasoner','r1qwen7b','openthinker3_7b']
 fig,axs=plt.subplots(1,3,figsize=(7.1,2.65),gridspec_kw={'wspace':.12})
 fig.subplots_adjust(left=.23,right=.99,top=.78,bottom=.25)
 for k,(ax,condition,label) in enumerate(zip(axs,['solve_mcq','generate','evaluate'],['A   Answer options','B   Generate approaches','C   Evaluate an answer'])):
  a=d[d.condition.eq(condition)].pivot(index='model',columns='label',values='delta').loc[models,B]*100
  im=heat(ax,a.to_numpy(),[model_label(m) for m in models] if k==0 else ['']*5,list('QPCRVBSK'),cmap=DIV,vmin=-15,vmax=15,fmt='+.0f',fontsize=6.8)
  ax.tick_params(axis='both',labelsize=7,pad=3);ax.set_title(label,loc='left',fontfamily='STIXGeneral',fontsize=9,pad=10)
 cax=fig.add_axes([.40,.08,.38,.025]);cb=colorbar(fig,im,cax=cax,orientation='horizontal',ticks=[-15,0,15]);cb.set_label('Change from free-response solving (pp)',fontsize=7,labelpad=1)
 save(fig,'figA6_demand_operations')

def readout():
 path=R/'readout/validation_summary.csv'
 if not path.exists():return
 d=pd.read_csv(path).query('domain=="all"')
 fig,axs=plt.subplots(1,2,figsize=(7.1,3.0),sharey=True);fig.subplots_adjust(left=.21,right=.98,top=.83,bottom=.25,wspace=.15)
 for ax,block,label in zip(axs,['future','next4'],['A   Future rates','B   Next-four rates']):
  for split,col,offset,marker,name in [('prompt_cv',BLUE,-.13,'o','Held-out prompts'),('heldout_domain',OCHRE,.13,'s','Held-out domains')]:
   a=d[d.split.eq(split)].set_index('target').loc[[block+'_'+b for b in B]]
   interval(ax,a.skill_over_reference,np.arange(8)+offset,a.lo,a.hi,col,marker,label=name)
  ax.set_yticks(range(8),L);ax.set_ylim(7.6,-.6);zero(ax);guides(ax);title(ax,label)
  ax.set_xlabel('MSE reduction over training reference')
 fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.02),ncol=2,frameon=False)
 save(fig,'figA7_readout_validation')

if __name__=='__main__':
 fidelity();similarity();demand();readout()
