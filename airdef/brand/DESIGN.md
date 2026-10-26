# Decart — Taste Index

Measured from decart.ai on 2026-10-27 at 1440×900: computed styles and the site's own `:root` custom properties on `/`, `/lucy`, `/about` and `/blog`. Tokens live in `brand.css`, behaviour in `brand.js`, and `index.html` renders every one of them.

**Legend** — **[M]** measured (computed style or the site's own CSS variable) · **[I]** inferred from a repeated pattern; judgment, not a site value · **[X]** invented to fill a gap; replace when you find the real value.

**Canonical source.** The home page and `/about` / `/blog` share one token set (CSS-module components, `--*-v2` variables). `/lucy` runs on an older set (`lucy-h1`, ink `#0d0d12`, grey `#6d6d6d`, 96px radii, Title Case buttons). Where they disagree, follow the home page.

---

## Brief

A white, quiet lab page where one electric blue does all the pointing and large, tightly tracked, regular-weight type does the talking.

## Principles

1. **One accent, used to point.** `#2548f6` marks the next action: the primary pill, "Read more →", the CTA band, the active blue tab. Everything else is greyscale. *Why:* the page is mostly white and grey, so a single saturated colour reads as "click here" without any other cue. If two blue things compete on one screen, demote one to a secondary outline pill. [M colour use / I rule]

2. **Weight comes from size, not boldness.** Every headline from 22px to 65px is weight 400. Weight 500 is for small UI text only: buttons, nav, tags, publication titles, the CTA-band title. Nothing on the canonical pages is 700. *Why:* thin-ish large type with tight tracking is what makes it feel like a research lab, not a SaaS landing page. [M]

3. **Tighter as it grows.** Tracking is −0.05em at display, −0.04em for headings, −0.02 to −0.03em for body and card titles, and only goes positive for tiny uppercase footer and legal text (+0.16em, +0.02em). *Why:* Roobert's default spacing is set for text sizes; at 48–65px it looks loose without this. [M]

4. **Soft containers, no lines.** Content groups sit in `#f5f5f5` fills with radii of 24–56px. There are no card borders and no card shadows. The only hairline is the dropdown's `1px #e2e2e4`; the only shadows are on things that float (dropdown, product UI frames). *Why:* fills give grouping without visual noise, and keep the blue the loudest thing on the page. [M]

5. **Everything you can press is a pill.** Buttons, nav links, tabs, tags and chips are all `border-radius: 1000px`. Small radii (8–16px) appear only on menu rows, icon boxes and the dropdown panel. *Why:* the shape alone tells you what's interactive. [M]

6. **Grey is the second voice.** `#8a8a8e` sets kickers ("Introducing"), dates, legal text and not-yet-revealed words; ink sets the claim. Hierarchy comes from grey vs ink, never from a second colour. It is the single most-used text colour on the home page (148 text nodes). [M]

7. **Quick and plain for controls, slow and continuous for atmosphere.** Every hover is 0.2s `ease` on colour/background/border. Ambient pieces (product-wash blobs, logo marquee) loop for 9–54s. Big statements reveal word by word as you scroll. No springs, no bounces, no hover lifts. [M durations / I framing]

---

## Color

| Role | Hex | Token (site var) | Where it's used | |
|---|---|---|---|---|
| Accent | `#2548f6` | `--dc-blue` (`--blue`) | Primary pill, CTA band, blue links, active blue tab, focus outline | M |
| Accent hover | `#0829cd` | `--dc-blue-hover` (`--medium-blue`) | Hover on primary pills and "Read more" | M |
| Accent pressed | `#061d94` | `--dc-blue-pressed` (`--dark-blue`) | Defined as pressed state; not observed | M |
| Blue tint | `#eff7ff` | `--dc-blue-tint` (`--chip-tint`) | Middle stop of the statement-section wash | M |
| Ink | `#000000` | `--dc-ink` | Hero display, secondary-pill outline, product tag | M |
| Ink 2 | `#101010` | `--dc-ink-2` | Section headings, card and publication titles | M |
| Dark | `#21222c` | `--dc-dark` | Inherited text colour; footer background | M |
| Body | `#333333` | `--dc-text-body` | Paragraphs; section ledes at 70% opacity | M |
| Sub | `#686868` | `--dc-text-sub` | Hero subheading | M |
| Nav | `#656565` | `--dc-text-nav` | Nav links at rest, dropdown descriptions | M |
| Grey | `#8a8a8e` | `--dc-grey` | Kickers, dates, legal, unrevealed text | M |
| Tab / pill / version greys | `#5d5d5d` / `#505050` / `#3a3a3a` | `--dc-text-tab` etc. | Inactive tab, hardware chip, "Version 2.5" | M |
| On blue | `#fdfcfc` | `--dc-on-blue` (`--almost-white`) | Text on primary pills | M |
| Page | `#ffffff` | `--dc-bg-page` | Every section, nav, icon boxes, active tab | M |
| Surface | `#f5f5f5` | `--dc-surface` (`--grey-card-2`) | Cards, media panels, tab tracks | M |
| Surface hover | `#f9f9f9` | `--dc-surface-hover` (`--grey-card`) | Hover fill for nav, ghost pill, dropdown rows | M |
| Chip | `#f6f6f6` | `--dc-surface-chip` | Hardware chips | M |
| Line | `#e2e2e4` | `--dc-line` | Dropdown border only | M |
| Body bg | `#f4f3f3` | `--dc-bg-body` | `<body>`; almost always hidden | M |
| Live | `#ff3b30` | `--dc-live` | Recording dot on live video, with a red glow | M |

**Washes** [M] — Lucy: `160deg, #dee8f9 → #c4e0fc 45% → #b3c3ea`, blob centres `#6b8ae8 #9fd0ff #cfd0ff`. Oasis: `#eef4e6 → #dde9d2 → #c9dbbb`, blobs `#86b56f #b9e0a5 #e6f2d2`. Statement: `#fff → #eff7ff 23% → #eff7ff 73% → #fff`.
*Why:* each product owns a hue family, but only inside its demo panel. UI chrome never picks those hues up.

**Not defined on the site:** error, warning, success, info, disabled. Don't invent them silently; if a feature needs one, add it to `brand.css` with an [X] tag and flag it. `#ff3b30` means "live", not "error".

**No dark theme.** Dark surfaces exist only as the footer (`#21222c`) and full-bleed video heroes (nav switches to transparent with white 85% links).

---

## Type

**Family.** Roobert by Displaay (commercial), weights 400 and 500 in use; 300 is loaded but unused. [M] The site also loads Monitor Light and Monitor Condensed and points `--font-heading` at Monitor, but no canonical page renders it — the current heading variable (`--font-heading-v2`) is Roobert. [M]
**Free substitute:** **Figtree** (Google Fonts, OFL), 400/500 — similar geometric grotesk proportions and a calm regular weight. [I] Keep the Roobert tracking values; Figtree sets slightly wider, so if headlines look loose, push display to −0.055em rather than changing weight. [I]

| Style | Size / line-height / tracking | Weight | Colour | Used for | |
|---|---|---|---|---|---|
| Display | 65px (4.0625rem) / 1.12 / −0.05em | 400 | `#000` | Home hero h1 (56px on /about) | M |
| Heading XL | 52px / 1.12 (1.24 in 2-col) / −0.04em | 400 | `#101010` | Section titles | M |
| Heading L | 48px / 1.24 / −0.04em | 400 | `#101010` | Model titles; statements at −0.05em / 1.2 | M |
| Heading M | 40px / 1.2 / −0.04em | 400 | `#101010` | Timeline items | M |
| Statement | 32px / 1.3 / −0.02em | 400 | `#000` | About-page running statement | M |
| CTA title | 32px / 1.4 / −0.03em | 500 | `#fff` | CTA band | M |
| Kicker | 24px / 1.2 / −0.02em | 400 | `#8a8a8e` | Label above a title | M |
| Card title | 22px / 1.4 / −0.03em | 400 | `#101010` | Feature cards | M |
| Pub title | 20px / 1.4 / −0.03em, clamp 3 lines | 500 | `#101010` | Publication cards | M |
| Body L | 18px / 1.52 / −0.02em | 400 | `#333` / `#686868` | Ledes, hero sub | M |
| Body | 16px / 1.52 / −0.02em | 400 | `#333` | Card body | M |
| UI | 16–18px / 1.5 / normal | 500 | varies | Buttons, nav, tabs | M |
| Meta | 14px / 1.4 / +0.01em | 400 | `#8a8a8e` | Dates | M |
| Caps | 13px / 1.4 / +0.16em, uppercase | 500 | `#fff` | Footer nav only | M |
| Legal | 11px / 2 / +0.02em, uppercase | 400 | `#8a8a8e` | Copyright, legal links | M |

Mobile sizes were not measured. `brand.css` scales headings down at 991px and 479px [X].

---

## Shape, spacing, layout

- **Radius** [M]: 1000px every control · 56px media panels · 40px feature cards · 24px CTA band and publication images · 16px dropdown · 12px dropdown rows and icon boxes · 8px dropdown icons. *Why:* radius grows with the container so nested shapes stay concentric-looking.
- **Borders** [M]: 1px pill outlines (`#000` secondary, `#fff` on dark, blue on primary) and the dropdown's `#e2e2e4`. Nothing else.
- **Shadows** [M]: dropdown `0 12px 32px rgba(33,34,44,.08)`; product UI frames inside panels `0 10px 58px rgba(0,0,0,.12–.14)`. Cards: none.
- **Container** [M]: 1360px (85rem) + 40px side padding = 1440px max. Sections: 80px top/bottom. Fixed white nav: 80px.
- **Grids** [M]: 3-up cards, gap 24px. Media + copy splits: 2 columns, gap 64px, alternating sides. Hero: `1fr | min(642px, 48.9%)`.
- **Inside cards** [M]: feature card padding 24/24/40, 48px gap between icon box (72px white square) and text. Publication card: image → 20px → text.
- **Breakpoints** [M]: 1199 / 991 / 767 / 479px (max-width). Nav collapses at 991.
- **Media panels** [M]: 1:1 or 1312:648 aspect, `#f5f5f5` or a product wash, product UI floats inside on a white frame. The wide tab panel has a white inverted-fillet notch cut into one corner holding the "Learn more" pill. [M]

---

## Components

**Pill button** [M] — 1px border, radius 1000, weight 500, line-height 1.5.
Sizes: `lg` 18px, padding 12/24 (53px tall) · `md` 18px, 52px tall, padding 0/32 · `sm` 16px, padding 8/19 (nav CTA).
Variants: `primary` blue fill, `#fdfcfc` text → `#0829cd` · `secondary` transparent, ink outline → fills ink, text white · `outline` white outline on dark/video → fills white, text ink · `white` white fill on the blue band → `#f9f9f9` · `ghost` nav grey → `#f9f9f9` fill, ink text.
Pairs are always primary + secondary (light) or primary + outline (dark). Disabled is not on the site [X: 40% opacity].

**Text link** [M] — `#2548f6`, "Read more →", hover `#0829cd`, no underline.

**Nav** [M] — fixed, white, 80px. Links are ghost pills (16px/500, `#656565`, padding 6/14/6/16, gap 6px); current page and open dropdown get the `#f9f9f9` fill. One `sm` primary CTA at the far right. Dropdown: white, 1px `#e2e2e4`, radius 16, padding 8, min-width 256, menu shadow; rows have a 36px rounded icon, 16px/500 label and 13px `#656565` description. It appears instantly; only the chevron animates. Over a dark hero the bar goes transparent with white 85% links and `rgba(255,255,255,.14)` hover.

**Segmented tabs** [M] — `#f5f5f5` pill track, 8px padding, 8px gap. Tab: 16px/500, padding 12/20, `#5d5d5d`; active is a white pill with `#333` text. Blue variant: active is blue with white text. Small variant: 15px, padding 8/16.

**Product tag + version** [M] — "Lucy" in a 1px ink-outlined pill (18px/500, padding 4/16), then "Version 2.5" in 16px/500 `#3a3a3a`. Sits above a model title.

**Hardware chip** [M] — `#f6f6f6`, `#505050`, 15px/500, padding 8/20/8/16, radius 52, logo icon left. Follows an inline label: "Built across:".

**Feature card** [M] — see Shape; 22px/400 title, 16px `#333` body.

**Publication card** [M] — 24px-radius image, 14px grey date, 20px/500 title clamped to 3 lines, 16px body, blue "Read more →". Hover: image scales 1.03 over 0.4s, link turns `#0829cd`. No card background.

**CTA band** [M] — blue, radius 24, padding 28.8/40, 32px/500 white title left, `white` pill right.

**Kicker + title + lede stack** [M] — grey 24px kicker, 20px gap, ink title, 18px lede at `#333`/0.7, then actions 32px below.

**Footer** [M] — `#21222c`, white. Uppercase 13px nav columns, hover opacity 0.7. Bottom row: 11px uppercase grey copyright left, legal links right, social icons centre.

**FAQ accordion** [M motion, I styling] — 18px/500 question, 16px `#6d6d6d` answer, opens with `grid-template-rows` over 0.35s `cubic-bezier(.39,.575,.565,1)`. Dividers in `brand.css` are [I].

**Logo strip** [M] — "Partners and customers" in 16px `#282828`, then one row of ~84×30 monochrome logos; on narrow screens it becomes a marquee.

---

## Motion

| What | Property | Duration | Easing | |
|---|---|---|---|---|
| All hovers | color, background-color, border-color | 0.2s | ease | M |
| Footer links | opacity → 0.7 | 0.2s | ease | M |
| Chevron | rotate 180° | 0.2s | ease | M |
| Dropdown panel | display only | 0 | — | M |
| Nav light ↔ dark | background, color | 0.3s | ease | M |
| Tab media | opacity crossfade | 0.35s | ease | M |
| Mobile menu | translateY(−100% → 0) | 0.35s | ease | M |
| Accordion | grid-template-rows / height | 0.35s | `cubic-bezier(.39,.575,.565,1)` | M |
| Card image | scale 1.03 | 0.4s | ease | M |
| Blurred backdrop | opacity | 0.6s | ease | M |
| Wash blobs | translate + scale loops; container spin | 9/11/13/16s; 54s | ease-in-out; linear | M |
| Live dot | opacity 1 ↔ 0.25 | not measured (1.2s in brand.css) | ease-in-out | M / X |
| Message in | opacity 0→1, translateY 5px→0 | not measured | — | M |
| Statement reveal | per-word opacity + colour, faint → `#8a8a8e` → ink, scroll-linked | — | linear | M look / I curve |
| Logo marquee | translateX 0 → −50% | not measured (40s in brand.css) | linear | M / X |

*Why it works:* controls respond instantly and quietly so the page feels engineered; the only "show" is the content itself (live video, the scroll reveal). The site honours `prefers-reduced-motion` by stopping the marquee [M]; `brand.css` extends that to all loops [I].

---

## Voice

- **Headlines** [M]: 4–8 words, present tense, declarative. "Edit reality as it happens." "What Live AI unlocks." Full-sentence claims end in a period: "One stack powers every layer of Live AI."
- **Product intros** [M]: grey kicker "Introducing" + product name with a period: "Decart Optimization Stack."
- **Recurring terms** [M]: "Live AI" (capitalised), "speed of reality", "live", "real time".
- **Numbers** [M]: concrete and typographic: "up to 100×", "30 FPS", "$300M".
- **Body** [M]: two sentences — claim, then short reframe. Triplets for punch: "No new racks. No new contracts. New ceiling."
- **Buttons** [M]: verb first, 1–3 words, sentence case on home: "Learn more", "Access demo", "Contact us", "Try Lucy 2.5", "See All" (one inconsistency). /lucy uses Title Case ("Talk To An Expert") — follow the home page [I].
- **Labels** [M]: "Partners and customers", "Built across:", "Version 2.5". Dates as "October 2026" or "July 16, 2026".

---

## Do / Don't

**Do**
- Use one primary blue pill per view; pair it with a secondary outline pill.
- Set headlines at 400 with negative tracking matched to size.
- Put a grey kicker above an ink title when a section needs a label.
- Group with `#f5f5f5` fills and radii from 24 to 56px.
- Keep sections white; use the blue-tint wash only behind a scroll-revealed statement.
- Keep hover changes to colour/background/border at 0.2s ease.
- Give each product its own wash inside its demo panel, and nowhere else.

**Don't**
- Bold large type (600/700) or add positive tracking above 13px.
- Put borders or shadows on cards.
- Use square or small-radius buttons, or mix pill and rectangle buttons in one row.
- Add a second UI accent; green and lavender belong to product washes only.
- Animate hover with lift, scale on buttons, or spring/bounce curves.
- Add a dark theme, uppercase headings, emoji or exclamation marks.
- Redraw the Decart mark or partner logos — use supplied SVGs.

---

## Uncertain or not covered

- **Mobile type scale** — breakpoints are measured, sizes at mobile are not; `brand.css` values are [X].
- **Scroll-reveal curve** — the faint → grey → ink sweep is what's on screen; the site drives it in JS and I couldn't read the timing function.
- **Marquee speed and live-dot timing** — keyframes measured, durations not.
- **Free font match** — Figtree is a judgment call; Roobert's quirkier details (R, t, a) aren't matched by any free face exactly.
- **Forms, inputs, error/success, tables, toasts, modals** — none on the pages measured (the cookie banner is third-party CookieYes and excluded).
- **Feature-card internal text gap** and **accordion dividers** — not measured.
