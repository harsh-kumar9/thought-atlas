"""Compact outcome and fidelity comparison of the frozen confirmation arms."""
from build_figures import *
R=P/'revision/results/confirmation'
d=pd.read_csv(R/'paired_effects.csv');b=pd.read_csv(R/'behavior/paired_shifts.csv')
fig,axs=plt.subplots(1,2,figsize=(7.1,2.9),gridspec_kw={'width_ratios':[1,1.05]})
fig.subplots_adjust(left=.13,right=.99,top=.81,bottom=.25,wspace=.78)
ax=axs[0]
for metric,offset,col,marker,label in [
    ('natural_correct',-.14,BLUE,'o','Natural (baseline 49.8%)'),
    ('elicited_correct',.14,OCHRE,'s','Elicited (baseline 95.1%)')]:
    a=d[d.metric.eq(metric)].set_index('condition').loc[['ol_math_A2','sig_moral']]
    interval(ax,a.delta*100,np.arange(2)+offset,a.lo*100,a.hi*100,col,marker,label=label)
zero(ax);guides(ax)
ax.set_yticks([0,1],['A2','Moral contrast']);ax.set_ylim(1.5,-.5)
ax.set_xlim(-9,41);ax.set_xticks([0,20,40]);ax.set_xlabel('Accuracy change (pp)')
title(ax,'A   Answer outcomes')
ax.text(0,1.02,'256 tasks; 8,192 total tokens',transform=ax.transAxes,fontsize=7,color=MUTED)
ax.legend(loc='upper left',bbox_to_anchor=(-.28,-.28),frameon=False,fontsize=7)
ax=axs[1]
for arm,offset,col,marker,label in [('ol_math_A2',-.14,BLUE,'o','A2'),('sig_moral',.14,OCHRE,'s','Moral contrast')]:
    a=b[b.condition.eq(arm)].set_index('target').loc[['future_'+x for x in B]]
    interval(ax,a.delta*100,np.arange(8)+offset,a.lo*100,a.hi*100,col,marker,label=label)
zero(ax);guides(ax)
ax.set_yticks(range(8),L,fontsize=7);ax.set_ylim(7.6,-.6)
ax.set_xlim(-8,10);ax.set_xticks([-5,0,5,10]);ax.set_xlabel('Future-operation rate change (pp)')
title(ax,'B   Achieved behavioral shifts')
ax.text(0,1.02,'Fixed 64-task subset; sample zero',transform=ax.transAxes,fontsize=7,color=MUTED)
ax.legend(loc='upper center',bbox_to_anchor=(.40,-.28),ncol=2,frameon=False,fontsize=7)
save(fig,'figA9_fresh_confirmation')
for name in ['paired_effects','endpoint_coverage','grade_sources']:
    shutil.copy2(R/f'{name}.csv',P/'supplement'/f'confirmation_{name}.csv')
shutil.copy2(R/'behavior/paired_shifts.csv',P/'supplement/confirmation_behavior_shifts.csv')
