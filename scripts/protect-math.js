/* global hexo */
'use strict';

// Keep $$...$$ and $...$ formulas away from the Markdown renderer
// (e.g. a lone "=" line inside $$ would otherwise become a heading),
// then put them back untouched inside .has-jax elements, which is what makes
// Stellar load MathJax on the page.

const escapeHtml = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const token = i => `MATHPROTECT${i}END`;
// Typora breaks lines on "\\" in $$ blocks; MathJax only does inside an environment.
const block = m => /\\\\/.test(m) && !/\\begin\{/.test(m)
  ? m.replace(/^\$\$([\s\S]*)\$\$$/, '$$$$\\begin{gathered}$1\\end{gathered}$$$$')
  : m;

hexo.extend.filter.register('before_post_render', data => {
  if (!data.content.includes('$')) return data;
  const saved = [];
  const keep = m => token(saved.push(m) - 1);
  // Split on fenced code blocks and inline code so formulas inside code stay as written.
  data.content = data.content
    .split(/(^```[\s\S]*?^```|^~~~[\s\S]*?^~~~|`[^`\n]+`)/m)
    .map((part, i) => i % 2 ? part : part
      .replace(/\$\$[\s\S]+?\$\$/g, m => `\n\n${keep(m)}\n\n`)
      .replace(/\$(?=\S)[^$\n]+?\S\$|\$\S\$/g, keep))
    .join('');
  data._mathSaved = saved;
  return data;
}, 1);

hexo.extend.filter.register('after_post_render', data => {
  const saved = data._mathSaved;
  if (!saved || !saved.length) return data;
  data.content = data.content
    .replace(new RegExp(`<p>${token('(\\d+)')}</p>`, 'g'), (_, i) => `<div class="has-jax">${escapeHtml(block(saved[i]))}</div>`)
    .replace(new RegExp(token('(\\d+)'), 'g'), (_, i) => `<span class="has-jax">${escapeHtml(saved[i])}</span>`);
  delete data._mathSaved;
  return data;
});
