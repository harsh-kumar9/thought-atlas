"""story_template.html -> story_template3.html: D-47 re-graded numbers + qualitative examples."""
S = open("story_template.html").read()
def rep(a, b):
    global S
    assert S.count(a) == 1, (S.count(a), a[:80])
    S = S.replace(a, b)

# ---------------- CSS for examples
rep("footer.end {", """.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 12px; align-items: start; }
.card { border: 1px solid var(--hair); border-radius: 8px; padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.card .ch { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }
.card .who { font-weight: 600; font-size: 0.95rem; }
.card .it { font-size: 0.92rem; font-weight: 500; }
.q { font-size: 0.88rem; color: var(--ink-2); max-width: 760px; }
.kw { font-size: 0.86rem; font-weight: 600; margin-top: 4px; }
pre.excerpt { font-family: var(--mono); font-size: 0.76rem; line-height: 1.55; color: var(--ink); background: var(--bg); border-radius: 6px; padding: 8px 10px; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.tl { display: flex; flex-direction: column; gap: 5px; }
.tag { order: -1; align-self: flex-start; font-size: 0.72rem; color: var(--ink-2); border: 1px solid var(--axis); border-radius: 999px; padding: 1px 8px; white-space: nowrap; }
.gap { font-size: 0.8rem; color: var(--muted); font-style: italic; }
.scores { display: flex; gap: 16px; font-size: 0.84rem; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.scores b { color: var(--ink); font-weight: 600; }
footer.end {""")

# ---------------- top summary
rep("""<p><strong>What steering buys is efficiency.</strong> On math, science and planning it reaches the same accuracy with a third to a half of the tokens, and the model almost never loops.</p>""",
    """<p><strong>What steering buys is efficiency.</strong> On math, science and planning it reaches what unsteered thinking gets with 32k tokens using a third to a half of the tokens, and the model almost never loops.</p>""")
rep("""<p><strong>RL finds the same behaviour.</strong> Training only a steering vector with RL lands on the same way of thinking, through an unrelated direction.</p>""",
    """<p><strong>RL finds the same behaviour.</strong> Training only a steering vector with RL lands on the same way of thinking, through an unrelated direction. It keeps checking its work and is the most accurate arm at 32k.</p>""")
rep("""<p>More accurate reasoning (when both finish, steered and unsteered answers are right equally often), better ideas, or new kinds of solutions from steering.</p>""",
    """<p>More accurate reasoning (on the problems it answers, A2 is right a little less often: 93% vs 97%), better ideas, or new kinds of solutions from steering.</p>""")
rep("""<div><dt>Held-out problems</dt>""", """<div><dt>Grading</dt><dd>A judge model reads the final answer and we compare it with the reference. We checked it by hand on 100 traces.</dd></div>
    <div><dt>Held-out problems</dt>""")

# ---------------- section 1 example
rep("""      <details class="nums"><summary>Show the numbers</summary><div class="tablewrap" data-table="sig"></div></details>
    </figure>""", """      <details class="nums"><summary>Show the numbers</summary><div class="tablewrap" data-table="sig"></div></details>
    </figure>
    <figure class="fig" id="x1">
      <div class="ftitle"><span class="fn">Example 1</span><h3>What the behaviours look like in a trace</h3></div>
      <div class="pair">
        <div class="card">
          <div class="ch">Math · 2,896 tokens</div>
          <p class="q">“Half the value of 3x − 9 is x + 37. What is the value of x?”</p>
          <div class="tl"><pre class="excerpt">Step 1: Eliminate the fraction. Multiply both sides of the equation by 2.
3x − 9 = 2(x + 37)
Step 2: Distribute the 2 on the right side. …
Step 4: Isolate x. …
x = 83</pre><span class="tag">subgoals</span></div>
          <div class="tl"><pre class="excerpt">4. Verify the Solution:
Plug x = 83 back into the original statement to see if it holds true. …
Does 120 = 120? Yes.</pre><span class="tag">checking</span></div>
          <div class="tl"><pre class="excerpt">Did I interpret "Half the value of" correctly? Yes, it means (3x − 9)/2. Did I make an arithmetic error? 2 × 37 = 74. …</pre><span class="tag">self-questioning</span></div>
        </div>
        <div class="card">
          <div class="ch">Idea generation · 5,705 tokens</div>
          <p class="q">Keyword: erosion. “Propose one novel, specific research idea related to the keyword below.”</p>
          <div class="tl"><pre class="excerpt">Idea 1 (Soil/Climate): AI-driven prediction of soil erosion under extreme precipitation scenarios. (A bit common, AI + climate is saturated). …
Idea 2 (Coastal): Bio-cementation of eroding coastlines using specific microorganisms. (Emerging, but maybe too applied). …
Idea 6 (Planetary Science): Erosion processes on Mars vs. Earth using new rover data. (Good, but maybe too speculative).</pre><span class="tag">new angle, then an objection</span></div>
          <div class="tl"><pre class="excerpt">Actually, let's look at Microplastic Erosion. …
Better: Permafrost Thaw and "Microbial Erosion" coupling.
Even better: The role of specific fungal networks (mycorrhizae) in preventing subsurface soil erosion under freeze-thaw cycles. …
Digital Erosion vs. Physical Erosion? No, stick to physical.
Idea: "Acoustic Levitation Erosion Testing." No.</pre><span class="tag">backtracking</span></div>
        </div>
      </div>
      <figcaption>Excerpts from two unsteered traces. “…” marks a cut and maths notation is simplified. The tags are our reading, to show what the behaviour names mean. The math trace works in steps and checks them; the idea trace keeps switching angle, objecting, and dropping ideas.</figcaption>
    </figure>""")

# ---------------- section 3
rep("""No push and no schedule improved both. On math, pushing toward exploring cost 13 points of accuracy.</p>""",
    """No push and no schedule improved both. On math, pushing toward exploring lowered accuracy.</p>""")
rep("""      <details class="nums"><summary>Show the numbers</summary><div class="tablewrap" data-table="trade"></div></details>
    </figure>""", """      <details class="nums"><summary>Show the numbers</summary><div class="tablewrap" data-table="trade"></div></details>
    </figure>
    <figure class="fig" id="x2">
      <div class="ftitle"><span class="fn">Example 2</span><h3>The same idea task, pushed each way</h3></div>
      <div class="kw">Keyword: fieldwork</div>
      <div class="pair">
        <div class="card">
          <div class="ch">Pushed toward exploring</div>
          <p class="it">“The ‘Stress-Strain’ Fieldwork Protocol: Quantifying Researcher Physiological Load vs. Sampling Accuracy in High-Stakes Ecological Monitoring”</p>
          <p class="q">Put wearables on field researchers and test whether stress makes the data they collect worse.</p>
          <div class="scores"><span>Originality <b>9</b></span><span>Feasibility <b>8</b></span></div>
        </div>
        <div class="card">
          <div class="ch">Pushed toward checking</div>
          <p class="it">“Quantifying the Observer Effect in Ecological Fieldwork: A Comparative Study of Nocturnal Insect Activity Recorded by Human Observers versus Autonomous Acoustic Sensors”</p>
          <p class="q">Count night insects with people and with sound recorders at the same place and time, and measure the difference.</p>
          <div class="scores"><span>Originality <b>8</b></span><span>Feasibility <b>9</b></span></div>
        </div>
      </div>
      <div class="kw">Keyword: educational psychology</div>
      <div class="pair">
        <div class="card">
          <div class="ch">Pushed toward exploring</div>
          <p class="it">“The ‘Transparency Paradox’ in AI Tutoring”</p>
          <p class="q">Test whether AI tutors that show their reasoning help students judge their own learning, even if students like them less.</p>
          <div class="scores"><span>Originality <b>9</b></span><span>Feasibility <b>8</b></span></div>
        </div>
        <div class="card">
          <div class="ch">Pushed toward checking</div>
          <p class="it">“The Impact of Explainable AI (XAI) Feedback on Metacognitive Regulation in Secondary Mathematics”</p>
          <p class="q">A randomised trial: students get plain right/wrong feedback or feedback that explains why, then are tested on fixing their own errors.</p>
          <div class="scores"><span>Originality <b>8</b></span><span>Feasibility <b>9</b></span></div>
        </div>
      </div>
      <figcaption>Final ideas for two of the 200 idea tasks, with the explore–check knob pushed each way. Titles are the model's own; the one-line summaries are ours. Scores are the judge's, 1 to 10. Exploring gives the bolder question, checking gives the study that is easier to run.</figcaption>
    </figure>""")

# ---------------- section 4
rep("""<p class="body prose">Steering toward A2 raised math accuracy at a 12k-token budget from 41% to 63% on 199 held-out problems. A random direction of the same size lowered it to 32%.</p>""",
    """<p class="body prose">Steering toward A2 raised math accuracy at a 12k-token budget from 57% to 81% on 199 held-out problems. The explore–check knob pushed toward checking reached 63%. A random direction of the same size lowered accuracy to 46%.</p>""")
rep("""max: 0.8, axis: true,
      ticks: [0, 0.2, 0.4, 0.6, 0.8], tfmt: pct""", """max: 1, axis: true,
      ticks: [0, 0.25, 0.5, 0.75, 1], tfmt: pct""")

# ---------------- section 5
rep("""<p class="body prose">Mostly faster. A2 at a 12k budget reaches the accuracy unsteered thinking needs 32k for, with 5.6k tokens on average instead of 13.2k. Its lead shrinks as the budget grows, from 24 points at 12k to 9 points at 32k. What remains at 32k comes from unsteered traces that never finish: 9% loop and 13% hit the cap. When both answer, they are equally accurate (72% and 71%). A2 also beats SEAL and the concise instruction.</p>""",
    """<p class="body prose">Mostly faster. With a 16k cap, A2 matches what unsteered thinking reaches with 32k (87% vs 85%), using 6.0k tokens on average instead of 13.2k. Its lead shrinks as the budget grows, from 27 points at 12k to 7 points at 32k. The lead that is left comes from finishing: at 32k, 12% of unsteered traces never answer (9% are stuck in a loop), against 1% for A2. On the problems each one answers, A2 is right a little less often (93% vs 97%), which fits it checking its work less. A2 beats SEAL by 22 points at 12k, and beats the concise instruction.</p>""")
rep("""    lineChart(box, { x: B, xticks: B, xfmt: b => b / 1024 + "k", xlabel: "token budget", ymin: 0, ymax: 0.8, yticks: [0, 0.2, 0.4, 0.6, 0.8], yfmt: pct,
      series: P1.map""", """    lineChart(box, { x: B, xticks: B, xfmt: b => b / 1024 + "k", xlabel: "token budget", ymin: 0, ymax: 1, yticks: [0, 0.25, 0.5, 0.75, 1], yfmt: pct,
      series: P1.map""")
rep("""annot: [{ x: 12288, y: DATA.math.ol_math_A2.correct[2], text: "A2 at 12k ≈ unsteered at 32k", anchor: "start" }] });""",
    """annot: [{ x: 16384, y: DATA.math.ol_math_A2.correct[3], text: "A2 at 16k ≈ unsteered at 32k", anchor: "start" }] });""")
rep("""table("p1", ["Arm"].concat(B.map(b => b / 1024 + "k")).concat(["Tokens (32k)", "Loops"]), P1.concat(["rl_zeroT_s60", "rl_a2initT_s60"]).map(k => [ENT[k].name].concat(DATA.math[k].correct.map(pct)).concat([kfmt(DATA.math[k].tokens[5]),""",
    """table("p1", ["Arm"].concat(B.map(b => b / 1024 + "k")).concat(["Answered (32k)", "Right when answered", "Tokens (32k)", "Loops"]), P1.concat(["rl_zeroT_s60", "rl_a2initT_s60"]).map(k => [ENT[k].name].concat(DATA.math[k].correct.map(pct)).concat([pct(DATA.math[k].answered[5]), pct(DATA.math[k].correct[5] / DATA.math[k].answered[5]), kfmt(DATA.math[k].tokens[5]),""")
rep("""      <figcaption>Same runs as Fig. 5.</figcaption>
    </figure>""", """      <figcaption>Same runs as Fig. 5.</figcaption>
    </figure>
    <figure class="fig" id="x3">
      <div class="ftitle"><span class="fn">Example 3</span><h3>One easy problem, three runs</h3></div>
      <p class="q">“The arithmetic mean of 7, 2, x and 10 is 9. What is the value of x?” (answer: 17)</p>
      <div class="chart" data-chart="same"></div>
      <div class="pair">
        <div class="card">
          <div class="ch">Unsteered, around token 450</div>
          <pre class="excerpt">Step 5c: Solve for x.
Subtract 19 from both sides.
x = 36 − 19
Step 5d: Calculate the result.
36 − 10 = 26
26 − 9 = 17
So, x = 17.</pre>
          <div class="gap">… and 32,000 tokens later, still thinking:</div>
          <pre class="excerpt">(Wait, I should check if I should mention the answer. Yes, clear answer.)
Okay.
I'll write it.
Done.
(Wait, I should check if I should mention the steps. Yes, clear steps.)
Okay.
I'll write it.
Done.
(Wait, I should check if I should mention the formula. Yes, clear</pre>
          <p class="q">“(Wait, …” appears 760 times. The budget runs out before the model answers.</p>
        </div>
        <div class="card">
          <div class="ch">A2, whole answer after 1,055 tokens</div>
          <pre class="excerpt">9 = (7 + 2 + x + 10) / 4
9 = (19 + x) / 4
36 = 19 + x
x = 36 − 19
x = 17
The value of x is 17.</pre>
          <p class="q">RL from zero writes the same steps after 877 tokens.</p>
        </div>
      </div>
      <div class="tablewrap"><table class="t">
        <tr><th>Problem (tokens used, seed 1)</th><th>Unsteered</th><th>A2</th><th>RL from zero</th></tr>
        <tr><td>Mean of 7, 2, x and 10 is 9; find x</td><td>32,768, no answer</td><td>1,055, right</td><td>877, right</td></tr>
        <tr><td>Battalions from 5 upper- and 10 lower-class soldiers</td><td>32,768, no answer</td><td>1,213, right</td><td>1,361, right</td></tr>
        <tr><td>Product of all x with |x − 4| − 10 = 2</td><td>32,768, no answer</td><td>1,623, right</td><td>1,371, right</td></tr>
        <tr><td>Assortments of 6 cookies from 3 kinds</td><td>32,768, no answer</td><td>1,629, right</td><td>4,829, right</td></tr>
      </table></div>
      <figcaption>This is the extreme case, not the typical one. 37 of 398 unsteered math traces got stuck in a loop like this; A2 solved 32 of those problems. Answers are shortened and maths notation is simplified.</figcaption>
    </figure>""")

# ---------------- section 6
rep("""The task's own vector does best. On GPQA it matches the unsteered 32k accuracy (74% vs 73%) with 5.2k tokens instead of 16.0k. The predictions got the order right.</p>""",
    """The task's own vector does best. On GPQA it reaches the unsteered 32k accuracy (76% vs 74%) with 5.2k tokens instead of 16.0k; on planning, 92% vs 89% with 4.0k tokens instead of 10.2k. The predictions got the order right.</p>""")

# ---------------- section 9
rep("""<p class="body prose">From zero, RL learns a vector that nearly matches A2 (59% vs 63% at 12k, 72% vs 71% at 32k, no loops). It is a third as long and points somewhere unrelated, but it changes the thinking the same way. Started from A2, RL keeps A2 and sharpens it: 13 points better than A2 at 4k, and 2k fewer tokens at 32k.</p>""",
    """<p class="body prose">From zero, RL learns a vector that matches A2 at 12k (79% vs 81%) and beats it at 32k (95% vs 92%), with no loops. It is a third as long and points somewhere unrelated, but it changes the thinking the same way, more gently, and with one difference: it does not cut checking. It finishes as often as A2 and is right as often as unsteered when it answers (97%). Started from A2, RL keeps A2 and sharpens it: 15 points better than A2 at 4k, and 2k fewer tokens at 32k.</p>""")
rep("""<span class="sub">share correct within 8k tokens, average of the last 6 steps</span>""", """<span class="sub">share correct within 8k tokens, as scored during training; average of the last 6 steps</span>""")
rep("""<figcaption>An earlier version of this experiment computed the training update at the wrong sampling temperature. Those runs are withdrawn; everything shown is after the fix.</figcaption>""",
    """<figcaption>The training reward was scored by our earlier grader, which missed about a quarter of correct math answers, so read its trend, not its level. The held-out accuracies are re-graded. An earlier version of this experiment computed the training update at the wrong sampling temperature; those runs are withdrawn and everything shown is after the fix.</figcaption>""")
rep("""    lineChart(box, { x: B, xticks: [4096, 8192, 16384, 24576, 32768], xfmt: b => b / 1024 + "k", xlabel: "token budget", ymin: 0, ymax: 0.8, yticks: [0, 0.2, 0.4, 0.6, 0.8], yfmt: pct,
      series: arms.map(k => ({ name: ENT[k].name, c: ENT[k].c, v: DATA.math[k].correct, label: lab[k] })), endLabels: true""",
    """    lineChart(box, { x: B, xticks: [4096, 8192, 16384, 24576, 32768], xfmt: b => b / 1024 + "k", xlabel: "token budget", ymin: 0, ymax: 1, yticks: [0, 0.25, 0.5, 0.75, 1], yfmt: pct,
      series: arms.map(k => ({ name: ENT[k].name, c: ENT[k].c, v: DATA.math[k].correct, label: lab[k] })), endLabels: true""")

# ---------------- section 10
rep("""Unprompted, the model uses the same method for a problem 93% of the time, and behaviour steering left the mix of methods unchanged.""",
    """Unprompted, the model uses the same method for a problem 95% of the time, and behaviour steering left the mix of methods unchanged.""")
rep("""<p class="body prose">Prompts can change the method. "Draw a picture" moves 31% of samples off the usual method, and a pool of half prompted, half unprompted samples solves more problems than either alone, the only such result in the project. Steering along that prompt's direction cannot: accuracy falls, loops rise, and the method barely changes.</p>""",
    """<p class="body prose">Prompts can change the method: "draw a picture" moves 29% of samples off the usual method. But a different method did not solve more problems. Mixing prompted and unprompted samples solves 86% of problems at least once in 4 tries, against 84% for unprompted samples alone, within noise. The prompt that did help, "work backwards" (67% to 76%), mostly kept the usual method and finished more often. Steering along the picture prompt's direction does worse: accuracy falls, loops rise, and the method barely changes.</p>""")
rep("""<div class="panel"><h4>Solved at least once in 4 samples</h4><span class="sub">pools of unprompted and "draw a picture" samples</span><div class="chart" data-chart="p3pool"></div></div>""",
    """<div class="panel"><h4>Solved at least once in 4 samples</h4><span class="sub">pools of 4 samples per problem</span><div class="chart" data-chart="p3pool"></div></div>""")
rep("""<figcaption>199 held-out math problems × 4 samples, 16k budget. "Usual method" is the method most unprompted samples use for that problem, as labelled by the judge.</figcaption>""",
    """<figcaption>199 held-out math problems × 4 samples, 16k budget. "Usual method" is the method most unprompted samples use for that problem, as labelled by the judge (with a tie, every tied method counts as usual). The mixed pool is 2 unprompted + 2 "draw a picture" samples: +1.7 points over unprompted, 95% interval −0.6 to +4.1.</figcaption>""")
rep("""    const p = DATA.p3.pools.figure;
    barChart(box, { rows: [["4 unprompted", p.unsteered], ["4 “draw a picture”", p.arm], ["2 + 2 mixed", p.mixed]]
      .map(([l, v], i) => ({ label: l, v, c: i === 2 ? "--bar-strong" : "--bar-neutral", strong: i === 2 })), min: 0, max: 1, vfmt: pct, what: "Solved at least once", labW: 150, rowH: 30 });""",
    """    const p = DATA.p3.pools;
    barChart(box, { rows: [["Unprompted", p.unprompted], ["“Draw a picture”", p.figure], ["2 + 2 mixed", p.mixed_figure], ["“Work backwards”", p.backward]]
      .map(([l, v], i) => ({ label: l, v, c: i === 2 ? "--bar-strong" : "--bar-neutral", strong: i === 2 })), min: 0, max: 1, vfmt: pct, what: "Solved at least once", labW: 150, rowH: 30 });""")
rep("""      <details class="nums"><summary>Show the numbers</summary><div class="tablewrap" data-table="p3"></div></details>
    </figure>""", """      <details class="nums"><summary>Show the numbers</summary><div class="tablewrap" data-table="p3"></div></details>
    </figure>
    <figure class="fig" id="x4">
      <div class="ftitle"><span class="fn">Example 4</span><h3>A new route is not always a better one</h3></div>
      <p class="q">“A 90° rotation around the origin in the counter-clockwise direction is applied to 7 + 2i. What is the resulting complex number?” (answer: −2 + 7i)</p>
      <div class="pair">
        <div class="card">
          <div class="ch">Unprompted · 4 samples</div>
          <p class="who">Multiplies by i</p>
          <pre class="excerpt">Specifically, multiplying by i corresponds to a 90° counter-clockwise rotation.</pre>
          <p class="q">All 4 samples use this method. 4 of 4 right, about 2,000 tokens each.</p>
        </div>
        <div class="card">
          <div class="ch">“Draw a picture” · 4 samples</div>
          <p class="who">Plots the point (7, 2) and turns it</p>
          <pre class="excerpt">Plot the initial point A = (7, 2).
Draw the vector from the origin to A.
Visualize the rotation. …
Plot the new point B = (−2, 7).</pre>
          <p class="q">All 4 samples use this method. 2 of 4 right (2.8k and 5.8k tokens); the other 2 ran out of the 16k budget.</p>
        </div>
      </div>
      <figcaption>Methods as labelled by the judge; excerpts from one sample each. On this problem the prompt changed the route every time and solved the problem less often. Across all 199 problems the picture prompt was 3 points less accurate than no prompt (within noise).</figcaption>
    </figure>""")

# ---------------- section 11
rep("""Reinforcement learning with a budgeted reward finds the same way of thinking on its own, through a different direction.</p>""",
    """Reinforcement learning with a budgeted reward finds the same way of thinking on its own, through a different direction, and the vector it finds is the most accurate we have at 32k.</p>""")
rep("""<p>Steering did not make the reasoning more accurate: when steered and unsteered traces both finish, they are right equally often, and the gain left at large budgets comes from avoiding loops. It did not make better ideas (it trades originality for feasibility), and it did not produce new kinds of solutions. A prompt can do the last one; its direction cannot.</p>""",
    """<p>Steering did not make the reasoning itself more accurate. A2 answers far more often but is right a little less often on what it answers (93% vs 97%); its gain at large budgets comes from finishing. It did not make better ideas (it trades originality for feasibility), and it did not produce new kinds of solutions. A prompt can change the method, but here that did not solve more problems, and steering along the prompt's direction does not change the method.</p>""")
rep("""<li>Most runs use one or two samples per task. Differences of a few points are within noise; the main effects are 10 to 36 points.</li>""",
    """<li>Most runs use one or two samples per task. Differences of a few points are within noise; the main effects are 15 to 38 points.</li>
        <li>An earlier grader missed about a quarter of correct math answers. Every math number on this page has been re-graded (checked by hand on 100 traces, all right), except the rank-one knob's math result in section 3, which is still being re-graded.</li>""")
rep("""decisions D-01 to D-45).""", """decisions D-01 to D-47).""")

# ---------------- keep trade-plot labels inside the plot on narrow screens
rep("""      txt(s, cx + o[0], cy + o[1], p.name, { anchor: o[2], size: 11.5, fill: "var(--ink)", weight: 500 });""",
    """      const lab = txt(s, cx + o[0], cy + o[1], p.name, { anchor: o[2], size: 11.5, fill: "var(--ink)", weight: 500 });
      const bb = lab.getBBox();
      if (bb.x + bb.width > W - 2) { lab.setAttribute("text-anchor", "end"); lab.setAttribute("x", cx - 8); }
      else if (bb.x < m.l + 2) { lab.setAttribute("text-anchor", "start"); lab.setAttribute("x", cx + 8); }""")

# ---------------- trade plot: numbered points + key on narrow screens
rep("""    for (const p of DATA.trade) {
      const cx = X(p.x[0]), cy = Y(p.y[0]), c = p.c === "a2" ? "--c-a2" : p.c === "random" ? "--c-random" : "--c-knob";""",
    """    const narrow = W < 520;
    for (const [i, p] of DATA.trade.entries()) {
      const cx = X(p.x[0]), cy = Y(p.y[0]), c = p.c === "a2" ? "--c-a2" : p.c === "random" ? "--c-random" : "--c-knob";""")
rep("""      const lab = txt(s, cx + o[0], cy + o[1], p.name, { anchor: o[2], size: 11.5, fill: "var(--ink)", weight: 500 });
      const bb = lab.getBBox();
      if (bb.x + bb.width > W - 2) { lab.setAttribute("text-anchor", "end"); lab.setAttribute("x", cx - 8); }
      else if (bb.x < m.l + 2) { lab.setAttribute("text-anchor", "start"); lab.setAttribute("x", cx + 8); }""",
    """      if (narrow) txt(s, cx + 7, cy - 11, String(i + 1), { size: 12, fill: "var(--ink)", weight: 600 });
      else {
        const lab = txt(s, cx + o[0], cy + o[1], p.name, { anchor: o[2], size: 11.5, fill: "var(--ink)", weight: 500 });
        const bb = lab.getBBox();
        if (bb.x + bb.width > W - 2) { lab.setAttribute("text-anchor", "end"); lab.setAttribute("x", cx - 8); }
      }""")
rep("""      hover(hit, p.name, [{ name: "Originality", value: `${fmt1(p.x[0])} (${fmt1(p.x[1])} to ${fmt1(p.x[2])})`, c }, { name: "Feasibility", value: `${fmt1(p.y[0])} (${fmt1(p.y[1])} to ${fmt1(p.y[2])})`, c }]);
    }""", """      hover(hit, p.name, [{ name: "Originality", value: `${fmt1(p.x[0])} (${fmt1(p.x[1])} to ${fmt1(p.x[2])})`, c }, { name: "Feasibility", value: `${fmt1(p.y[0])} (${fmt1(p.y[1])} to ${fmt1(p.y[2])})`, c }]);
    }
    if (narrow) {
      const key = document.createElement("ol"); key.className = "tkey";
      for (const p of DATA.trade) { const li = document.createElement("li"); li.textContent = p.name; key.appendChild(li); }
      box.appendChild(key);
    }""")
rep("""footer.end {""", """ol.tkey { margin: 8px 0 0 0; padding-left: 1.4em; font-size: 0.84rem; color: var(--ink-2); display: grid; gap: 2px; }
footer.end {""")

# ---------------- chart for example 3
rep("""  // ---------------------------------------------------------------- tables (static)""", """  CH.same = box => {
    const W = Math.max(280, box.clientWidth), labW = Math.min(112, W * 0.3), m = { l: labW, r: 12, t: 8, b: 22 };
    const rows = [{ label: "Unsteered", v: 32768, c: "--c-base", h: 58 }, { label: "A2", v: 1055, c: "--c-a2", h: 34 }, { label: "RL from zero", v: 877, c: "--c-rlzero", h: 34 }];
    const H = m.t + m.b + rows.reduce((a, r) => a + r.h, 0), s = svgFor(box, W, H), iw = W - m.l - m.r, X = v => m.l + v / 32768 * iw;
    for (const t of [0, 8192, 16384, 24576, 32768]) {
      el("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, style: { stroke: t === 0 ? "var(--axis)" : "var(--hair)", strokeWidth: 1 } }, s);
      txt(s, X(t), H - m.b + 13, t ? t / 1024 + "k" : "0", { anchor: t === 32768 ? "end" : "middle", size: 10.5 });
    }
    let y = m.t;
    for (const r of rows) {
      const yc = y + (r.label === "Unsteered" ? 20 : r.h / 2), bh = 14, x1 = X(r.v), rr = Math.min(4, x1 - m.l);
      txt(s, labW - 10, yc, r.label, { anchor: "end", size: 12, fill: "var(--ink-2)" });
      el("path", { d: `M${m.l},${yc - bh / 2}H${x1 - rr}Q${x1},${yc - bh / 2} ${x1},${yc - bh / 2 + rr}V${yc + bh / 2 - rr}Q${x1},${yc + bh / 2} ${x1 - rr},${yc + bh / 2}H${m.l}Z`, style: { fill: `var(${r.c})` } }, s);
      if (r.label === "Unsteered") {
        const xm = X(452);
        el("line", { x1: xm, x2: xm, y1: yc - bh / 2 - 3, y2: yc + bh / 2 + 5, style: { stroke: "var(--ink)", strokeWidth: 1.5 } }, s);
        txt(s, xm + 4, yc + bh / 2 + 13, "x = 17 written at token 452", { size: 11, fill: "var(--ink)" });
        txt(s, X(32768), yc - bh / 2 - 8, "32,768 tokens, no answer", { anchor: "end", size: 11, fill: "var(--ink)" });
      } else txt(s, x1 + 6, yc, `${r.v.toLocaleString("en-US")} tokens, answers 17`, { size: 11.5, fill: "var(--ink)" });
      const hit = el("rect", { x: 0, y, width: W, height: r.h, style: { fill: "transparent" } }, s);
      hover(hit, r.label, [{ name: "Tokens", value: r.v.toLocaleString("en-US"), c: r.c }, { name: "Answer", value: r.label === "Unsteered" ? "none (budget ran out)" : "17 (right)" }]);
      y += r.h;
    }
  };

  // ---------------------------------------------------------------- tables (static)""")

open("story_template3.html", "w").write(S)
data = open("story_compact3.json").read()
open("thought-atlas-story.html", "w").write(S.replace("/*__DATA__*/", data))
print("ok", len(S))
