"""All fixed-checkpoint RL seeds, with completion beside accuracy."""
from build_figures import *

R = P/'revision/results/rl_evaluation'
assert (R/'aggregate_verification.json').exists()
d = pd.read_csv(R/'paired_effects.csv')
runs = [f'{init}_seed{s}' for init in ['zero', 'ol_math_A2'] for s in range(3)]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
fig.subplots_adjust(left=.19, right=.98, top=.78, bottom=.24, wspace=.28)
for ax, reference, heading in zip(axes, ['baseline', 'a2'], ['A   Change from unsteered', 'B   Change from fixed A2']):
    for metric, offset, color, marker, label in [('success', -.13, BLUE, 'o', 'Accuracy'), ('completed', .13, OCHRE, 's', 'Completion')]:
        q = d[d.reference.eq(reference) & d.metric.eq(metric)].set_index('run').loc[runs]
        interval(ax, q.delta*100, np.arange(6)+offset, q.lo*100, q.hi*100, color, marker, label=label)
    zero(ax); guides(ax); ax.axhline(2.5, color=RULE, lw=.6)
    ax.set_xlabel('Paired difference (pp)')
    ax.text(0, 1.20, heading, transform=ax.transAxes, fontsize=10, fontfamily='DejaVu Serif')
    base = d[d.reference.eq(reference)].groupby('metric').reference_absolute.first()
    ax.text(0, 1.06, f'Reference: {base["success"]*100:.1f}% accurate · {base["completed"]*100:.1f}% complete',
            transform=ax.transAxes, fontsize=6.7, color=MUTED)
axes[0].set_yticks(range(6), ['Zero start · seed 0', 'Zero start · seed 1', 'Zero start · seed 2',
                            'A2 start · seed 0', 'A2 start · seed 1', 'A2 start · seed 2'], fontsize=7)
axes[0].set_ylim(5.6, -.6)
axes[0].legend(loc='upper left', bbox_to_anchor=(0, -.25), ncol=2, frameon=False, fontsize=7)
save(fig, 'figA11_clean_rl')
for name in ['paired_effects', 'absolute_by_training_seed', 'fallback_sensitivity']:
    shutil.copy2(R/f'{name}.csv', P/'supplement'/f'clean_rl_{name}.csv')
