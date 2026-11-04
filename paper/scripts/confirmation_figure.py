"""Compact comparison of frozen fresh-item endpoints; all prespecified arms."""
from build_figures import *
R=P/'revision/results/confirmation'
d=pd.read_csv(R/'paired_effects.csv')
fig,ax=plt.subplots(figsize=(3.35,2.05))
fig.subplots_adjust(left=.28,right=.97,top=.88,bottom=.35)
for metric,offset,col,marker,label in [
    ('natural_correct',-.14,BLUE,'o','Natural completion (baseline 49.8%)'),
    ('elicited_correct',.14,OCHRE,'s','Answer elicitation (baseline 95.1%)')]:
    a=d[d.metric.eq(metric)].set_index('condition').loc[['ol_math_A2','sig_moral']]
    interval(ax,a.delta*100,np.arange(2)+offset,a.lo*100,a.hi*100,col,marker,label=label)
zero(ax);guides(ax)
ax.set_yticks([0,1],['A2','Moral contrast']);ax.set_ylim(1.5,-.5)
ax.set_xlim(-9,41);ax.set_xticks([0,10,20,30,40]);ax.set_xlabel('Accuracy change (percentage points)')
ax.set_title('256 fresh project items · 8,192 tokens',loc='left',fontsize=8.5,pad=8)
fig.legend(*ax.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.12,-.015),frameon=False,fontsize=7)
save(fig,'figA9_fresh_confirmation')
for name in ['paired_effects','endpoint_coverage','grade_sources']:
    shutil.copy2(R/f'{name}.csv',P/'supplement'/f'confirmation_{name}.csv')
