"""Inline-SVG chart builders for the results report.

Colors are emitted as CSS custom properties (--series-N, --grid, --text-*) so the
page can restep them for dark mode without touching the geometry.
"""

import math

W, H = 360, 250
PAD_L, PAD_R, PAD_T, PAD_B = 52, 14, 26, 40


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _log_ticks(lo, hi):
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    step = 1 if (b - a) <= 8 else (2 if (b - a) <= 16 else 3)
    return [10.0 ** e for e in range(int(a), int(b) + 1, step)]


def _fmt_pow(v):
    e = int(round(math.log10(v)))
    sup = str(e).replace("-", "−")
    return f"10{_sup(sup)}"


_SUPS = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
         "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
         "−": "⁻"}


def _sup(s):
    return "".join(_SUPS.get(c, c) for c in s)


class LogPanel:
    """One small-multiple panel: log x, log y."""

    def __init__(self, title, xlo, xhi, ylo, yhi, xlabel="", ylabel="", note=None):
        self.title, self.note = title, note
        self.xlo, self.xhi = xlo, xhi
        self.ylo, self.yhi = max(ylo, 1e-18), yhi
        self.xlabel, self.ylabel = xlabel, ylabel
        self.series = []
        self.hlines = []

    def add(self, label, xs, ys, slot, dashed=False):
        pts = [(x, y) for x, y in zip(xs, ys)
               if x and y and y == y and y > 0 and math.isfinite(y)]
        if pts:
            self.series.append((label, pts, slot, dashed))

    def hline(self, y, label):
        self.hlines.append((y, label))

    def _sx(self, x):
        t = (math.log10(x) - math.log10(self.xlo)) / (math.log10(self.xhi) - math.log10(self.xlo))
        return PAD_L + t * (W - PAD_L - PAD_R)

    def _sy(self, y):
        y = min(max(y, self.ylo), self.yhi)
        t = (math.log10(y) - math.log10(self.ylo)) / (math.log10(self.yhi) - math.log10(self.ylo))
        return H - PAD_B - t * (H - PAD_T - PAD_B)

    def svg(self):
        o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="{_esc(self.title)}">']
        o.append(f'<text x="{PAD_L - 42}" y="14" class="ttl">{_esc(self.title)}</text>')
        if self.note:
            o.append(f'<text x="{W - PAD_R}" y="14" class="note" text-anchor="end">'
                     f'{_esc(self.note)}</text>')
        # grid + axes
        for yt in _log_ticks(self.ylo, self.yhi):
            if not (self.ylo <= yt <= self.yhi):
                continue
            y = self._sy(yt)
            o.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" class="grid"/>')
            o.append(f'<text x="{PAD_L-6}" y="{y+3.5:.1f}" class="tick" text-anchor="end">'
                     f'{_fmt_pow(yt)}</text>')
        for xt in _log_ticks(self.xlo, self.xhi):
            if not (self.xlo <= xt <= self.xhi):
                continue
            x = self._sx(xt)
            o.append(f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{H-PAD_B}" class="grid"/>')
            o.append(f'<text x="{x:.1f}" y="{H-PAD_B+15}" class="tick" text-anchor="middle">'
                     f'{_fmt_pow(xt)}</text>')
        o.append(f'<line x1="{PAD_L}" y1="{H-PAD_B}" x2="{W-PAD_R}" y2="{H-PAD_B}" class="axis"/>')
        o.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H-PAD_B}" class="axis"/>')
        for yv, lab in self.hlines:
            if self.ylo <= yv <= self.yhi:
                y = self._sy(yv)
                o.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                         f'class="ref"/>')
                o.append(f'<text x="{W-PAD_R-3}" y="{y-4:.1f}" class="note" '
                         f'text-anchor="end">{_esc(lab)}</text>')
        # series
        for label, pts, slot, dashed in self.series:
            dd = " ".join(f'{"M" if i == 0 else "L"}{self._sx(x):.1f},{self._sy(y):.1f}'
                          for i, (x, y) in enumerate(pts))
            dash = ' stroke-dasharray="5 3"' if dashed else ""
            o.append(f'<path d="{dd}" fill="none" stroke="var(--series-{slot})" '
                     f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
            for x, y in pts:
                o.append(f'<circle cx="{self._sx(x):.1f}" cy="{self._sy(y):.1f}" r="4" '
                         f'fill="var(--series-{slot})" stroke="var(--surface-1)" '
                         f'stroke-width="2"><title>{_esc(label)}: h={x:.0e}, '
                         f'{y:.3e}</title></circle>')
        if self.xlabel:
            o.append(f'<text x="{(PAD_L+W-PAD_R)/2:.0f}" y="{H-4}" class="axlab" '
                     f'text-anchor="middle">{_esc(self.xlabel)}</text>')
        if self.ylabel:
            o.append(f'<text transform="translate(11,{(PAD_T+H-PAD_B)/2:.0f}) rotate(-90)" '
                     f'class="axlab" text-anchor="middle">{_esc(self.ylabel)}</text>')
        o.append("</svg>")
        return "".join(o)


def legend(items):
    """items: [(label, slot, dashed)]"""
    parts = ['<div class="legend">']
    for lab, slot, dashed in items:
        st = "border-top:2px dashed" if dashed else "border-top:2px solid"
        parts.append(f'<span class="lg"><i style="{st} var(--series-{slot})"></i>'
                     f'{_esc(lab)}</span>')
    parts.append("</div>")
    return "".join(parts)


def grid(panels, cols=3):
    return (f'<div class="grid" style="--cols:{cols}">' +
            "".join(f'<figure class="panel">{p}</figure>' for p in panels) + "</div>")


# ------------------------------------------------------------------ bar chart
def bar_chart(title, groups, series, values, ylabel="", logy=True, fmt="{:.1f}"):
    """groups: x category labels; series: [(name, slot)]; values[s][g]."""
    BW, BH = 620, 280
    pl, pr, pt, pb = 62, 16, 30, 46
    flat = [v for row in values for v in row if v and v > 0]
    if not flat:
        return ""
    lo, hi = min(flat), max(flat)
    lo = lo / 3
    hi = hi * 2

    def sy(v):
        if logy:
            t = (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        else:
            t = v / hi
        return BH - pb - t * (BH - pt - pb)

    o = [f'<svg viewBox="0 0 {BW} {BH}" class="chart" role="img" aria-label="{_esc(title)}">']
    o.append(f'<text x="20" y="16" class="ttl">{_esc(title)}</text>')
    for yt in _log_ticks(lo, hi):
        if not (lo <= yt <= hi):
            continue
        y = sy(yt)
        o.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{BW-pr}" y2="{y:.1f}" class="grid"/>')
        o.append(f'<text x="{pl-6}" y="{y+3.5:.1f}" class="tick" text-anchor="end">'
                 f'{_fmt_pow(yt)}</text>')
    gw = (BW - pl - pr) / len(groups)
    n = len(series)
    bw = min(30.0, (gw - 14) / n)
    for gi, g in enumerate(groups):
        gx = pl + gi * gw
        for si, (sname, slot) in enumerate(series):
            v = values[si][gi]
            if not v or v <= 0:
                continue
            x = gx + gw / 2 - (n * bw + (n - 1) * 2) / 2 + si * (bw + 2)
            y = sy(v)
            o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                     f'height="{BH-pb-y:.1f}" rx="4" fill="var(--series-{slot})">'
                     f'<title>{_esc(sname)} / {_esc(g)}: {v:.4g}</title></rect>')
            o.append(f'<text x="{x+bw/2:.1f}" y="{y-5:.1f}" class="blab" '
                     f'text-anchor="middle">{fmt.format(v)}</text>')
        o.append(f'<text x="{gx+gw/2:.1f}" y="{BH-pb+16}" class="tick" '
                 f'text-anchor="middle">{_esc(g)}</text>')
    o.append(f'<line x1="{pl}" y1="{BH-pb}" x2="{BW-pr}" y2="{BH-pb}" class="axis"/>')
    if ylabel:
        o.append(f'<text transform="translate(13,{(pt+BH-pb)/2:.0f}) rotate(-90)" '
                 f'class="axlab" text-anchor="middle">{_esc(ylabel)}</text>')
    o.append("</svg>")
    return "".join(o)
