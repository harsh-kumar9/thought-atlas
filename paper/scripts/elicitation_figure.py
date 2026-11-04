"""Plot frozen matched-total-budget estimates; no fitting or cell selection."""
from build_figures import *

R = P / 'revision/results/elicitation'
d = pd.read_csv(R / 'paired.csv')
arms = ['sig_math', 'sig_gpqa', 'sig_idea', 'ol_random']
labels = ['Math contrast', 'GPQA contrast', 'Idea contrast', 'Random vector']
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.65), sharey=True)
fig.subplots_adjust(left=.17, right=.99, top=.72, bottom=.28, wspace=.20)
for ax, target, letter in zip(axs, ['math', 'gpqa', 'planning'], 'ABC'):
    for budget, offset, col, marker, name in [
        (12288, -.13, BLUE, 'o', '12,288 tokens'),
        (32768, .13, OCHRE, 's', '32,768 tokens'),
    ]:
        a = d[d.task_type.eq(target) & d.budget.eq(budget) & d.metric.eq('correct')].set_index('condition').loc[arms]
        interval(ax, a.delta * 100, np.arange(4) + offset, a.lo * 100, a.hi * 100, col, marker, label=name)
    zero(ax); guides(ax)
    ax.set_yticks(range(4), labels); ax.set_ylim(3.5, -.5)
    ax.set_xlim(-23, 7); ax.set_xticks([-20, -10, 0])
    ax.set_xlabel('Accuracy change (pp)')
    base = d[d.task_type.eq(target) & d.metric.eq('correct')].groupby('budget').baseline.first()
    title(ax, f'{letter}   {domain(target)}')
    ax.text(0, 1.04, f'Baseline {100*base[12288]:.1f}% / {100*base[32768]:.1f}%',
            transform=ax.transAxes, fontsize=7, color=MUTED)
fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center',
           bbox_to_anchor=(.57, .01), ncol=2, frameon=False)
save(fig, 'figA8_matched_elicitation')
for name in ['paired', 'summary', 'grade_sources']:
    shutil.copy2(R / f'{name}.csv', P / 'supplement' / f'elicitation_{name}.csv')
