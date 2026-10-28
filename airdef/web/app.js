// Typical air-defence demo. Three altitudes: the decision for one threat (panel), its story over time (flight strip),
// tonight's outcome (Tonight block). One accent (brand blue) points at the focal threat and the call; all else greyscale.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const money = (m) => (m < 0.001 ? "<$1k" : m < 1 ? `$${Math.round(m * 1000)}k` : m < 1000 ? `$${m.toFixed(1)}M` : `$${(m / 1000).toFixed(2)}B`);
const BLUE = [37, 72, 246], GREY = [138, 138, 142], INK = [16, 16, 16], HIT = [255, 59, 48];
const VIEWS = { theatre: [40.0, 31.9, 5.1, 30], north: [35.35, 32.85, 8.3, 40], south: [39.5, 24.5, 4.8, 30] };
const SCEN_VIEW = { northern_barrage: "north", april_mixed: "theatre", october_saturation: "theatre", june_attrition: "theatre", houthi_eilat: "south" };
const DRILL_VIEW = { wait_for_cheaper: "theatre", last_chance: "north", let_it_fall: "north", track_not_firm: "theatre", coverage_gap: "south", helicopter: "theatre" };
const viewOf = (s) => (s.startsWith("drill:") ? DRILL_VIEW[s.split(":")[1]] : SCEN_VIEW[s]);
const DONE = ["killed", "leaked", "fell_open"], ENGAGED = ["assigned", "inflight"];

// ---------------------------------------------------------------- copy (EN / HE)
const I18N = {
  en: {
    speed: "Speed", speedDesc: ["Real time", "Calm walkthrough", "A raid in a few minutes", "Fast-forward"],
    brand: "Air defence", skip: "Skip ahead", settings: "Settings", lang: "עברית", play: "Play", pause: "Pause", deciding: "Deciding…",
    skipped: (x) => `Skipped ${x} of quiet`,
    cls: { B: "Ballistic missile", R: "Rocket", C: "Cruise missile", D: "Drone" },
    to: "to", openGround: "Open ground near", priority: "priority", impactIn: "Impact in", follow: "Follow the most urgent",
    following: "In focus",
    act: { hold: "Let it fall", wait: "Wait", cheaper: "Wait for cheaper", air: { fighter: "Fighter jets", helicopter: "Helicopter" } },
    whyAir: (u, m, pk) => `${u} can meet it in ${m} and ${pk}.`,
    whyTrack: "The track is not firm yet. Deciding later keeps every layer that still reaches it open.",
    local: { fires: (u, n) => `${u} fires ${n}`, holds: (u) => `${u} holds`, left: (n) => `${n} left`, busy: "all launch channels are busy", reload: "its launchers are reloading",
      friendly: "own aircraft are near the intercept point", debris: (s) => `a later intercept keeps the debris off ${s}`,
      last: "the last chance to engage it", nolook: "no time for a second look", look: "time to look before a second shot" },
    pk: ["is almost certain to stop it", "is very likely to stop it", "is likely to stop it", "may stop it", "is unlikely to stop it"],
    whyFire: (e, pk) => `${e} can engage now and ${pk}.`,
    whyFire2: (e, pk) => `Two ${e} together: the pair ${pk}.`,
    whyWait: (e, m, pk) => `${e} reaches it in ${m} and ${pk}.`,
    whyHoldOpen: "It will land in open ground. Nothing worth an interceptor is at risk.",
    whyHold: "The damage it can do is worth less than any interceptor that can reach it.",
    vs: { raid: (r) => `${r} would spend interceptors tonight's raid still needs.`, campaign: (r) => `${r} would spend stock the campaign has to keep.`,
      local: (r, d) => `${r} costs ${d} more for this threat, in interceptors and risk.` },
    status: { killed: "Intercepted", leaked: "Hit a defended site", fell_open: "Fell in open ground" },
    sec: { t: "Threats", u: "Aircraft", b: "Fire units" }, options: "Options weighed",
    st: { out: "Tracking", wait: "Wait", cheaper: (e) => `Wait for ${e}`, hold: "Let it fall", ground: (e, u) => `${e} · ${u}`,
      air: (u) => `${u} en route`, fired: (u, n) => `${u} fired ${n}`, hit: (s) => `Hit ${s}` },
    bat: { ready: "Ready", engaging: (n) => `Engaging ${n}`, reloading: "Reloading", empty: "Empty", unlimited: "Unlimited",
      onLaunchers: "Ready on launchers", left: "Rounds left", channels: "Channels in use", reload: "Next reload", firedTonight: "Fired tonight",
      targets: "Engaging", none: "—", sub: (r, l) => `${r} ready · ${l} left` },
    unit: { ground: "On alert", cap: "On patrol", tasked: "Intercepting", rtb: "Returning to base", rearm: "Rearming",
      weapons: "Weapons left", task: "Task", state: "State" },
    legend: { engaged: "Engaged", waiting: "Waiting", fall: "Let it fall", unit: "Fire unit", air: "Aircraft" },
    prog: "How the call developed", shotHit: (e) => `✓ ${e} intercepted it`, shotMiss: (e) => `✕ ${e} missed`,
    intro: "Every threat, one question: which interceptor, and when.", pressPlay: "Press Play.",
    seeking: "On to the next moment", raidOver: "Raid over", recap: "Tonight's moments", restart: "Restart", backStory: "Back to the story",
    shotsLine: (e, n) => `${e} fired ${n}; missed.`, noShot: "Nothing could reach it in time.",
    inAir: (n, total) => `${n} in the air · ${total}`, mags: "Magazines",
    held: (f, c, k) => `Held back: ${f} let fall · ${c} waited for a cheaper interceptor · ${k} kept`,
    markShort: { engage: "Engage", air: "Aircraft", cheaper: "Wait", payoff: "Paid off", letfall: "Fall", holdfire: "Hold", hit: "Hit" },
    raidLine: "Raid", threatLine: "This threat",
    next: "Next moment", modes: { general: "General", focus: "Focus" }, showAll: "Show everything",
    mom: { engage: "Engagement", air: "Aircraft sent", cheaper: "Wait for cheaper", payoff: "The wait paid off", letfall: "Salvo let fall",
      holdfire: "Holding fire", hit: "Hit" },
    letfall: (n) => `${n} land in open ground. No interceptor spent on them.`,
    salvoOf: (c, o) => `${c} from ${o}`,
    holdfire: { friendly: (u) => `${u} holds fire: own aircraft are near the intercept point.`, debris: (u, s) => `${u} holds fire: the debris would fall on ${s}.` },
    hitLine: (x, s) => `${x} hit ${s}.`, payoffLine: (e) => `${e} takes it now, as planned.`,
    clsPl: { B: "ballistic missiles", R: "rockets", C: "cruise missiles", D: "drones" },
    origin: { iran_w: "western Iran", iran_nw: "north-western Iran", iran_c: "central Iran", iraq: "Iraq", yemen: "Yemen",
      lebanon_s: "southern Lebanon", lebanon_bq: "the Beqaa", gaza: "Gaza" },
    salvo: (n, c, o) => `${n} ${c} from ${o}`,
    tally: { fall: (n) => `${n} let fall`, engaged: (n) => `${n} engaged`, killed: (n) => `${n} intercepted`, hits: (n) => `${n} hit` },
    intercepted: "Intercepted",
    reads: "What Typical reads", notYet: "No interceptor can reach it yet. Typical decides once one can.",
    empty: "Press Play. The most urgent threat and Typical's call appear here.",
    drills: "Drills", drill: { kicker: (n, k) => `Drill · case ${n} of ${k}`, played: "Each option played out to the end, 40 times. Average cost:",
      best: "best", next: "Next case", thinking: "Typical is deciding…", who: { typical: "Typical", planner: "Planner" },
      fam: { wait_for_cheaper: "Wait for cheaper", last_chance: "Last chance", let_it_fall: "Let it fall", track_not_firm: "Unclear track", coverage_gap: "Coverage gap", helicopter: "Helicopter" },
      blurb: { wait_for_cheaper: "Arrow 3 can fire now; a cheaper layer reaches it later, and Arrow 3 is scarce.",
        last_chance: "Seconds before impact, one battery still reaches it.", let_it_fall: "The rocket will land well clear of any defended site.",
        track_not_firm: "Early in flight, the predicted impact straddles the edge of a city.",
        coverage_gap: "A drone heads for Eilat, the Eilat battery is empty, and aircraft can still meet it.",
        helicopter: "A drone heads north; the batteries are empty, the laser is clouded out, the fighters are busy." } },
    tonight: "Tonight", attacker: (x) => `Attacker spent ${x}`, spent: "Spent", hits: "Hits on sites", a3: "Arrow 3 left",
    details: "Details",
    planner: "Planner",
    strip: (id, t, s) => `Flight of ${id} · ${t} to ${s}`, now: "now", impact: "impact", calls: "Calls", hit: "hit", miss: "miss",
    posture: "Magazine posture", postures: ["Healthy", "Conserve", "Critical"], prio: "Priority sites", assets: "Assets and conditions",
    allied: "US THAAD and Navy", weather: "Cloud cover (laser degraded)", knobs: "Model and sensors", pkScale: "Pk scale", escTop: "Escalate below",
    warn: "Early warning, min", dmg: "Damage weights, $M per hit", tier: { strategic: "Strategic site", city: "City", infra: "Infrastructure" },
    prioMult: "Priority ×", warhead: "Warhead weight (1 = 500 kg)", perSite: "Value per site (blank = by type)",
    note: "Posture, stock, damage weights and priorities feed every lens. Typical reads them as sentences; the planner reads numbers.",
  },
  he: {
    speed: "מהירות", speedDesc: ["זמן אמת", "הדגמה רגועה", "מתקפה בכמה דקות", "הרצה מהירה"],
    brand: "הגנה אווירית", skip: "דלג קדימה", settings: "הגדרות", lang: "English", play: "הפעל", pause: "השהה", deciding: "מחליט…",
    skipped: (x) => `דולגו ${x} שקטות`,
    cls: { B: "טיל בליסטי", R: "רקטה", C: "טיל שיוט", D: "כטב״ם" },
    to: "לעבר", openGround: "שטח פתוח ליד", priority: "בעדיפות", impactIn: "פגיעה בעוד", follow: "עקוב אחרי הדחוף",
    following: "בפוקוס",
    act: { hold: "לתת לו ליפול", wait: "להמתין", cheaper: "להמתין לזול יותר", air: { fighter: "מטוסי קרב", helicopter: "מסוק" } },
    whyAir: (u, m, pk) => `${u} יכולים לפגוש אותו בעוד ${m}, ו${pk}.`,
    whyTrack: "המסלול עוד לא יציב. החלטה מאוחרת יותר שומרת את כל השכבות שעוד מגיעות אליו.",
    local: { fires: (u, n) => `${u} יורה ${n}`, holds: (u) => `${u} ממתין`, left: (n) => `נותרו ${n}`, busy: "כל ערוצי השיגור תפוסים", reload: "המשגרים בטעינה",
      friendly: "מטוסים שלנו קרובים לנקודת היירוט", debris: (s) => `יירוט מאוחר יותר ירחיק את הרסיסים מ${s}`,
      last: "ההזדמנות האחרונה", nolook: "אין זמן לבדוק לפני ירי נוסף", look: "יש זמן לבדוק לפני ירי נוסף" },
    pk: ["כמעט בטוח שיעצור אותו", "סביר מאוד שיעצור אותו", "סביר שיעצור אותו", "עשוי לעצור אותו", "לא סביר שיעצור אותו"],
    whyFire: (e, pk) => `${e} יכול ליירט עכשיו, ו${pk}.`,
    whyFire2: (e, pk) => `שני ${e} יחד: ${pk}.`,
    whyWait: (e, m, pk) => `${e} יגיע אליו בעוד ${m}, ו${pk}.`,
    whyHoldOpen: "הוא ייפול בשטח פתוח. אין שם דבר ששווה מיירט.",
    whyHold: "הנזק שהוא יכול לגרום שווה פחות מכל מיירט שמגיע אליו.",
    vs: { raid: (r) => `${r} ינצל מיירטים שהמתקפה הלילה עוד צריכה.`, campaign: (r) => `${r} ינצל מלאי שהמערכה חייבת לשמור.`,
      local: (r, d) => `${r} עולה ${d} יותר על האיום הזה, במיירטים ובסיכון.` },
    status: { killed: "יורט", leaked: "פגע באתר מוגן", fell_open: "נפל בשטח פתוח" },
    sec: { t: "איומים", u: "כלי טיס", b: "יחידות אש" }, options: "האפשרויות שנשקלו",
    st: { out: "במעקב", wait: "להמתין", cheaper: (e) => `להמתין ל${e}`, hold: "לתת לו ליפול", ground: (e, u) => `${e} · ${u}`,
      air: (u) => `${u} בדרך`, fired: (u, n) => `${u} ירה ${n}`, hit: (s) => `פגע ב${s}` },
    bat: { ready: "מוכנה", engaging: (n) => `מיירטת ${n}`, reloading: "בטעינה", empty: "ריקה", unlimited: "ללא הגבלה",
      onLaunchers: "מוכנים במשגרים", left: "מיירטים שנותרו", channels: "ערוצים בשימוש", reload: "טעינה הבאה", firedTonight: "נורו הלילה",
      targets: "מיירטת את", none: "—", sub: (r, l) => `${r} מוכנים · ${l} נותרו` },
    unit: { ground: "בכוננות", cap: "בסיור", tasked: "ביירוט", rtb: "חוזר לבסיס", rearm: "בחימוש",
      weapons: "חימוש שנותר", task: "משימה", state: "מצב" },
    legend: { engaged: "ביירוט", waiting: "בהמתנה", fall: "ליפול", unit: "יחידת אש", air: "כלי טיס" },
    prog: "איך ההחלטה התפתחה", shotHit: (e) => `✓ ${e} יירט אותו`, shotMiss: (e) => `✕ ${e} החטיא`,
    intro: "לכל איום, שאלה אחת: איזה מיירט, ומתי.", pressPlay: "לחצו הפעל.",
    seeking: "ממשיכים לרגע הבא", raidOver: "המתקפה הסתיימה", recap: "הרגעים של הלילה", restart: "מההתחלה", backStory: "חזרה לסיפור",
    shotsLine: (e, n) => `${e} ירה ${n}; החטיא.`, noShot: "שום מיירט לא הגיע אליו בזמן.",
    inAir: (n, total) => `${n} באוויר · ${total}`, mags: "מלאים",
    held: (f, c, k) => `נחסך: ${f} נפלו בשטח פתוח · ${c} המתינו למיירט זול יותר · ${k} נשמרו`,
    markShort: { engage: "יירוט", air: "כלי טיס", cheaper: "המתנה", payoff: "השתלם", letfall: "נפילה", holdfire: "עצירה", hit: "פגיעה" },
    raidLine: "המתקפה", threatLine: "האיום",
    next: "הרגע הבא", modes: { general: "תמונה כללית", focus: "מיקוד" }, showAll: "הצג הכול",
    mom: { engage: "יירוט", air: "שיגור כלי טיס", cheaper: "להמתין לזול יותר", payoff: "ההמתנה השתלמה", letfall: "מטח שנופל בשטח פתוח",
      holdfire: "עצירת אש", hit: "פגיעה" },
    letfall: (n) => `${n} נופלים בשטח פתוח. אף מיירט לא נורה עליהם.`,
    salvoOf: (c, o) => `${c} מ${o}`,
    holdfire: { friendly: (u) => `${u} עוצר אש: מטוסים שלנו קרובים לנקודת היירוט.`, debris: (u, s) => `${u} עוצר אש: הרסיסים היו נופלים על ${s}.` },
    hitLine: (x, s) => `${x} פגע ב${s}.`, payoffLine: (e) => `${e} מיירט אותו עכשיו, כמתוכנן.`,
    clsPl: { B: "טילים בליסטיים", R: "רקטות", C: "טילי שיוט", D: "כטב״מים" },
    origin: { iran_w: "מערב איראן", iran_nw: "צפון-מערב איראן", iran_c: "מרכז איראן", iraq: "עיראק", yemen: "תימן",
      lebanon_s: "דרום לבנון", lebanon_bq: "הבקעה", gaza: "עזה" },
    salvo: (n, c, o) => `${n} ${c} מ${o}`,
    tally: { fall: (n) => `${n} נופלים`, engaged: (n) => `${n} ביירוט`, killed: (n) => `${n} יורטו`, hits: (n) => `${n} פגעו` },
    intercepted: "יורטו",
    reads: "מה Typical קורא (באנגלית)", notYet: "אף מיירט עדיין לא מגיע אליו. Typical יחליט כשיגיע.",
    empty: "לחצו הפעל. האיום הדחוף ביותר וההחלטה של Typical יופיעו כאן.",
    drills: "תרגולים", drill: { kicker: (n, k) => `תרגול · מקרה ${n} מתוך ${k}`, played: "כל אפשרות הורצה עד הסוף, 40 פעמים. עלות ממוצעת:",
      best: "הטוב ביותר", next: "המקרה הבא", thinking: "Typical מחליט…", who: { typical: "Typical", planner: "מתכנן" },
      fam: { wait_for_cheaper: "להמתין לזול יותר", last_chance: "הזדמנות אחרונה", let_it_fall: "לתת לו ליפול", track_not_firm: "מסלול לא ברור", coverage_gap: "פער כיסוי", helicopter: "מסוק" },
      blurb: { wait_for_cheaper: "חץ 3 יכול לירות עכשיו; שכבה זולה יותר תגיע אליו מאוחר יותר, וחץ 3 במחסור.",
        last_chance: "שניות לפני הפגיעה, סוללה אחת עדיין מגיעה אליו.", let_it_fall: "הרקטה תיפול הרחק מכל אתר מוגן.",
        track_not_firm: "בתחילת המעוף, נקודת הפגיעה החזויה על גבול עיר.",
        coverage_gap: "כטב״ם בדרך לאילת, הסוללה באילת ריקה, וכלי טיס עוד יכולים לפגוש אותו.",
        helicopter: "כטב״ם בדרך לצפון; הסוללות ריקות, הלייזר מוגבל בעננים, מטוסי הקרב תפוסים." } },
    tonight: "הלילה", attacker: (x) => `התוקף הוציא ${x}`, spent: "הוצאה", hits: "פגיעות באתרים", a3: "חץ 3 שנותרו",
    details: "פרטים",
    planner: "מתכנן",
    strip: (id, t, s) => `מסלול ${id} · ${t} לעבר ${s}`, now: "עכשיו", impact: "פגיעה", calls: "החלטות", hit: "פגע", miss: "החטיא",
    posture: "מצב מלאים", postures: ["תקין", "שימור", "קריטי"], prio: "אתרים בעדיפות", assets: "נכסים ותנאים",
    allied: "THAAD וצי ארה״ב", weather: "עננות (לייזר מוגבל)", knobs: "מודל וחיישנים", pkScale: "מקדם Pk", escTop: "למפעיל מתחת",
    warn: "התרעה מוקדמת, דק׳", dmg: "משקלי נזק, מיליון $ לפגיעה", tier: { strategic: "אתר אסטרטגי", city: "עיר", infra: "תשתית" },
    prioMult: "מכפיל עדיפות", warhead: "משקל ראש קרב (1 = 500 ק״ג)", perSite: "ערך לאתר (ריק = לפי סוג)",
    note: "מצב המלאים, המלאי, משקלי הנזק והעדיפויות נכנסים לכל עדשה. Typical קורא אותם כמשפטים; המתכנן כמספרים.",
  },
};
let LANG = localStorage.getItem("airdef-lang") || "en";
const T = () => I18N[LANG];
const nm = (o) => (LANG === "he" && o.name_he ? o.name_he : o.name);
const glyphs = (s) => (/[\u0590-\u05FF]/.test(s) ? [...s].reverse().join("") : s);   // deck.gl TextLayer has no bidi: flip Hebrew only
const effName = (e) => nm(META.effectors[e]).replace("F-15/F-35 ", "").replace("F-16 laser-guided rocket (APKWS)", "APKWS").replace("GPS jamming (EW)", "EW jamming").replace(" laser", "");
const actLabel = (key) => {
  if (!key) return "—";
  if (key in T().act && typeof T().act[key] === "string") return T().act[key];
  const [a, x] = key.split(":");
  return a === "air" ? T().act.air[x] : effName(x);
};
const unitName = (id) => { const u = (S?.units || []).find((x) => x.id === id); return u ? nm(u) : id; };
const batEff = (id) => META.batteries.find((b) => b.id === id)?.eff;
const siteName = (sid) => { const s = META.sites.find((x) => x.id === sid); return s ? nm(s) : ""; };
const fmtT = (s) => { const neg = s < 0; s = Math.abs(s); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = Math.floor(s % 60); return (neg ? "−" : "") + (h ? h + ":" : "") + String(m).padStart(2, "0") + ":" + String(x).padStart(2, "0"); };
const mins = (s) => (LANG === "he" ? (s < 90 ? `${Math.round(s)} שניות` : `${Math.round(s / 60)} דקות`) : (s < 90 ? `${Math.round(s)} seconds` : `${Math.round(s / 60)} minutes`));
const pkIdx = (p) => (p >= 0.95 ? 0 : p >= 0.85 ? 1 : p >= 0.7 ? 2 : p >= 0.5 ? 3 : 4);

let META, PATHS = {}, PATH_SCEN = null, S = null, fetchedAt = 0, pinned = false, detail = null;
let focus = null;            // the expanded object: {k: "t" threat | "b" fire unit | "u" aircraft, id}
let selected = null;         // the threat on the map and the flight line: the focused threat, or the focused unit's target
const collapsed = {};        // list sections the viewer folded
const openGroups = new Set(); // salvos the viewer opened
let caption = null;          // the moment on the caption card (stays while paused)
const isFocus = (k, id) => focus && focus.k === k && focus.id === id;
const viewPol = "typical";
const flashSeen = new Map();

// ---------------------------------------------------------------- map
const map = new maplibregl.Map({ container: "map", style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  center: [35.0, 31.6], zoom: 6.6, pitch: 45, maxPitch: 75, attributionControl: { compact: true } });
const overlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
map.on("load", () => {   // the basemap is a quiet canvas: no labels of its own (we label countries and defended sites), grey borders
  for (const l of map.getStyle().layers) {
    if (l.type === "symbol") map.setLayoutProperty(l.id, "visibility", "none");
    else if (l.id.startsWith("boundary")) map.setPaintProperty(l.id, "line-color", "#cfcfd3");
  }
  map.setPaintProperty("water", "fill-color", "#e6e9eb");
});
const COUNTRIES = [["Lebanon", "לבנון", 34.05, 35.95], ["Syria", "סוריה", 34.9, 38.4], ["Jordan", "ירדן", 30.9, 36.9], ["Iraq", "עיראק", 32.8, 43.6],
  ["Iran", "איראן", 32.4, 53.2], ["Saudi Arabia", "ערב הסעודית", 25.2, 44.2], ["Yemen", "תימן", 15.6, 47.2], ["Egypt", "מצרים", 27.2, 30.6],
  ["Israel", "ישראל", 30.75, 34.85], ["Cyprus", "קפריסין", 35.1, 33.2], ["Turkey", "טורקיה", 38.6, 35.2]].map(([en, he, lat, lon]) => ({ name: en, name_he: he, lat, lon }));
const ATLAS = (() => {   // two mask icons: a square (fire unit) and a chevron (aircraft); colour comes from the layer
  const c = document.createElement("canvas"); c.width = 128; c.height = 64; const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(6, 6, 52, 52);
  g.beginPath(); g.moveTo(96, 4); g.lineTo(122, 60); g.lineTo(96, 46); g.lineTo(70, 60); g.closePath(); g.fill();
  return c.toDataURL();
})();
const ICONS = { sq: { x: 0, y: 0, width: 64, height: 64, mask: true }, tri: { x: 64, y: 0, width: 64, height: 64, mask: true } };
const LABEL = { fontFamily: "Figtree, Heebo, Arial, sans-serif", characterSet: "auto", outlineWidth: 3, outlineColor: [255, 255, 255, 235],
  fontSettings: { sdf: true }, extensions: [new deck.CollisionFilterExtension()], collisionGroup: "labels" };
const bearing = (u) => (u.goal ? (Math.atan2((u.goal[1] - u.lon) * Math.cos(u.lat * Math.PI / 180), u.goal[0] - u.lat) * 180) / Math.PI : 0);
map.addControl(overlay);
const flyTo = (v) => { const [lon, lat, zoom, pitch] = VIEWS[v]; map.flyTo({ center: [lon, lat], zoom, pitch, duration: 1600 }); };
const api = (p, body) => fetch(p, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {}).then((r) => r.json());

async function init() {
  META = await api("/api/meta");
  $("#lang").onclick = () => { LANG = LANG === "en" ? "he" : "en"; localStorage.setItem("airdef-lang", LANG); applyLang(); };
  $("#scenBtn").onclick = (e) => { e.stopPropagation(); $("#scenDrop").classList.toggle("is-open"); };
  $("#speedBtn").onclick = (e) => { e.stopPropagation(); $("#speedDrop").classList.toggle("is-open"); };
  document.addEventListener("click", () => document.querySelectorAll(".dc-dropdown").forEach((d) => d.classList.remove("is-open")));
  $("#speedMenu").onclick = (e) => { const x = e.target.closest("[data-x]")?.dataset.x; if (x) api("/api/control", { action: "speed", speed: +x }); };
  $("#scenMenu").onclick = async (e) => {
    const s = e.target.closest("[data-s]")?.dataset.s; if (!s) return;
    focus = selected = null; pinned = false; detail = null; detailOpen = false; await api("/api/control", { action: "reset", scenario: s }); flyTo(viewOf(s));
  };
  $("#play").onclick = () => api("/api/control", raidOver() ? { action: "reset" } : { action: S?.running ? "pause" : "play" });
  $("#next").onclick = () => { pinned = false; api("/api/control", { action: "next" }); };
  $("#mode").onclick = (e) => { const b = e.target.closest("[data-mode]"); if (b) setMode(b.dataset.mode); };
  $("#raid").onclick = (e) => {   // a moment on the timeline: replay it, in Focus
    const m = e.target.closest("[data-m]"); if (!m) return;
    if (S.mode !== "focus") setMode("focus");
    pinned = false; detail = null; api("/api/control", { action: "moment", index: +m.dataset.m });
  };
  $("#gear").onclick = () => { const o = $("#sheet").classList.toggle("open"); $("#gear").setAttribute("aria-expanded", o); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { $("#sheet").classList.remove("open"); $("#gear").setAttribute("aria-expanded", false); } });
  $("#list").onclick = (e) => {
    if (e.target.closest("#followBtn")) { pinned = false; focus = null; renderList(); return; }
    if (e.target.closest("#nextCase")) { focus = selected = null; pinned = false; detail = null; api("/api/control", { action: "reset", scenario: `drill:${S.drill.family}:${S.drill.case + 1}` }); return; }
    const mm = e.target.closest("[data-m]"); if (mm) { pinned = false; detail = null; api("/api/control", { action: "moment", index: +mm.dataset.m }); return; }
    const md = e.target.closest("[data-mode]"); if (md) { setMode(md.dataset.mode); return; }
    const sec = e.target.closest("[data-sec]"); if (sec) { collapsed[sec.dataset.sec] = !collapsed[sec.dataset.sec]; renderList(); return; }
    const g = e.target.closest("[data-g]"); if (g) { openGroups.has(g.dataset.g) ? openGroups.delete(g.dataset.g) : openGroups.add(g.dataset.g); renderList(); return; }
    const o = e.target.closest("[data-k]"); if (o && !e.target.closest("details")) select(o.dataset.k, o.dataset.id);
  };
  await poll();
  applyLang();
  flyTo(viewOf(S.scenario));
  setInterval(poll, 250);
  setInterval(loadDetail, 400);
  requestAnimationFrame(frame);
}

function applyLang() {
  document.documentElement.lang = LANG;
  document.documentElement.dir = LANG === "he" ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = T()[el.dataset.i18n]));
  const lg = T().legend;
  $("#legend").innerHTML = `<span><i class="dot ink"></i>${lg.engaged}</span><span><i class="dot hollow"></i>${lg.waiting}</span><span><i class="dot faint"></i>${lg.fall}</span>
    <span><i class="sq"></i>${lg.unit}</span><span><i class="tri"></i>${lg.air}</span><span><i class="dot blue"></i>${T().following}</span>`;
  $("#lang").textContent = T().lang;
  $("#scenMenu").innerHTML = Object.entries(META.scenarios).map(([k, v]) => {
    const [title, blurb] = LANG === "he" ? [v.title_he, v.blurb_he] : [v.title, v.blurb];
    return `<button class="dc-dropdown__item" data-s="${k}"><span class="dc-dropdown__text"><span class="dc-dropdown__label">${esc(title)}</span><span class="dc-dropdown__desc">${esc(blurb)}</span></span></button>`;
  }).join("") + `<p class="dc-dropdown__desc menuhead">${T().drills}</p>` + Object.entries(T().drill.fam).map(([k, title]) =>
    `<button class="dc-dropdown__item" data-s="drill:${k}:0"><span class="dc-dropdown__text"><span class="dc-dropdown__label">${esc(title)}</span><span class="dc-dropdown__desc">${esc(T().drill.blurb[k])}</span></span></button>`).join("");
  renderBar(); renderTonight(); renderList(); renderStrip(); buildSheet();
}

async function poll() {
  try { S = await api(`/api/state?policy=${viewPol}`); fetchedAt = performance.now(); } catch { return; }
  if (S.scenario !== PATH_SCEN) {
    const P = await api("/api/paths");
    PATHS = Object.fromEntries(P.threats.map((t) => [t.id, t]));
    PATH_SCEN = P.scenario; flashSeen.clear(); openGroups.clear(); caption = null; firedAt = {};
    if (S.drill) setTimeout(() => flyToThreat(S.drill.threat), 300);
  }
  const story = S.mode === "focus", beat = story && S.beat && S.moments?.[S.beat.id];
  if (beat && caption?.id !== beat.id) { pinned = false; flyToMoment(beat); }   // a new moment always takes the panel
  if (beat) caption = beat;                            // and stays (greyed) until the next one
  if (!story || (S.moments || []).length === 0 && !beat) caption = story ? caption : null;
  if (S.drill && !pinned) focus = { k: "t", id: S.drill.threat };
  else if (story && !pinned) focus = caption ? { k: "t", id: caption.tid } : null;
  else if (!story && !pinned) focus = null;            // General: nothing is singled out until the viewer clicks
  syncSelected();
  renderBar(); renderTonight(); renderList(); renderRaid();
}

let scrollToFocus = false;
function setMode(m) {
  pinned = false; focus = null; caption = null; renderList.last = null;
  api("/api/control", { action: "mode", mode: m });
  if (S) { S.mode = m; if (m === "general") flyTo(viewOf(S.scenario)); renderBar(); renderList(); }
}
const raidOver = () => S && !S.drill && S.t >= S.end - 0.5;
const gkey = (tr) => `${PATHS[tr.id]?.origin}|${tr.cls}|${Math.floor(tr.t0 / 180)}`;   // a salvo: same origin, class, 3 minutes
const salvoOf = (m) => { const tr = S.tracks.find((x) => x.id === m.tid), g = tr && gkey(tr);
  return g ? S.tracks.filter((x) => gkey(x) === g && (x.status === "held" || x.status === "fell_open")) : []; };
function fitTo(points, cls) {   // fit the camera to a set of [lon, lat] points; zoom capped per threat class
  const lons = points.map((p) => p[0]), lats = points.map((p) => p[1]);
  // frame from straight above, then tilt with a little less zoom: a pitched fitBounds pushes the far side off screen
  const cam = map.cameraForBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]],
    { padding: { top: 100, bottom: 110, left: 100, right: 100 }, maxZoom: { R: 9, D: 8 }[cls] ?? 6.5 });
  if (cam) map.flyTo({ center: cam.center, zoom: cam.zoom - 0.35, pitch: 30, bearing: 0, duration: 1400 });
}
function flyToThreat(tid) {
  const path = PATHS[tid]; if (!path) return;
  fitTo([posAt(path, S.t).slice(0, 2), path.p.at(-1).slice(0, 2)], path.cls);
}
function flyToMoment(m) {   // Focus: frame the decision, not just the threat -- its head, where it will land, and who takes it
  const path = PATHS[m.tid]; if (!path) return;
  const pts = [posAt(path, S.t).slice(0, 2)], tr = S.tracks.find((x) => x.id === m.tid);
  if (tr) pts.push([tr.ipp[0], tr.ipp[1]]);
  if (m.type === "letfall") for (const x of salvoOf(m)) pts.push(posAt(PATHS[x.id], S.t).slice(0, 2));
  const waited = m.info.opts.find((o) => o[0] === "cheaper")?.[6];
  for (const id of [tr?.unit, m.unit, waited].filter(Boolean)) {
    const b = META.batteries.find((x) => x.id === id), u = S.units.find((x) => x.id === id);
    if (b) pts.push([b.lon, b.lat]); if (u) pts.push([u.lon, u.lat]);
  }
  fitTo(pts, path.cls);
}
function objectPos(k, id) {
  if (k === "t") { const p = PATHS[id]; return p && posAt(p, tNow).slice(0, 2); }
  const o = k === "b" ? META.batteries.find((x) => x.id === id) : S.units.find((x) => x.id === id);
  return o && [o.lon, o.lat];
}
function select(k, id) {   // click a row or a map object: expand it; click the expanded one again: let go
  if (isFocus(k, id)) { pinned = false; focus = null; }
  else {
    focus = { k, id }; pinned = true;
    const at = objectPos(k, id);
    if (at && !map.getBounds().contains(at)) map.easeTo({ center: at, duration: 900 });   // the selected object is on screen
  }
  syncSelected(); detail = null; loadDetail(); renderList();
}
function syncSelected() {
  const prev = selected;
  if (!focus) selected = null;
  else if (focus.k === "t") selected = focus.id;
  else if (focus.k === "b") selected = (S.batteries.find((b) => b.id === focus.id)?.tasks || [])[0] || null;
  else selected = S.units.find((u) => u.id === focus.id)?.task || null;
  if (selected !== prev) loadDetail();
}
async function loadDetail() {
  if (!selected) { detail = null; renderList(); renderStrip(); return; }
  try { detail = await api(`/api/track/${selected}?policy=${viewPol}`); } catch { return; }
  const u = detail.assign?.unit;
  if (S.mode === "focus" && caption?.tid === detail.id && u && u !== loadDetail.unit) flyToMoment({ ...caption, unit: u });   // the camera takes in who takes it
  loadDetail.unit = u;
  renderList(); renderStrip();
}

// One display clock per frame: eases toward the server's (extrapolated) time, never runs backwards, snaps only on
// real jumps (auto-skip, reset). Polls arrive every 250 ms with server time that wobbles; showing it raw jitters.
let shown = null, lastFrame = performance.now(), tNow = 0;
function simNow() {
  if (!S) return 0;
  const now = performance.now(), dtr = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  const live = S.running;
  const rate = S.rate ?? S.speed;   // the playback rate actually running (a story slows it, fast-forward raises it)
  const target = live ? Math.min(S.end, S.t + ((now - fetchedAt) / 1000) * rate) : S.t;
  if (shown === null || !S.running || Math.abs(target - shown) > Math.max(30, 4 * rate)) return (shown = target);
  const step = live ? dtr * rate : 0;
  shown = Math.max(shown, shown + step + (target - shown - step) * Math.min(1, dtr * 3));
  return shown;
}
function posAt(path, t) {
  const p = path.p;
  if (t <= p[0][3]) return p[0];
  if (t >= p[p.length - 1][3]) return p[p.length - 1];
  let lo = 0, hi = p.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (p[m][3] <= t) lo = m; else hi = m; }
  const a = p[lo], b = p[hi], f = (t - a[3]) / (b[3] - a[3]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, t];
}

// ---------------------------------------------------------------- map layers (greyscale; the focal threat in blue)
function frame() {
  requestAnimationFrame(frame);
  if (!S || !META) return;
  const t = (tNow = simNow()), prio = new Set(S.cfg.priority_sites), dim = S.mode === "focus" && !!caption;
  const hl = S.mode === "focus" && caption?.type === "letfall" ? new Set(salvoOf(caption).map((x) => x.id)) : null;   // the whole salvo
  const live = [], trails = [], futures = [], falls = [];
  for (const tr of S.tracks) {
    const path = PATHS[tr.id]; if (!path) continue;
    if (t > Math.min(S.killed[tr.id] ?? Infinity, path.t0 + path.T)) continue;
    const held = tr.status === "held", sel = tr.id === selected || !!hl?.has(tr.id);
    const cur = posAt(path, t), col = sel ? BLUE : held ? [170, 170, 174] : GREY;
    const past = path.p.filter((q) => q[3] < t).map((q) => q.slice(0, 3)); past.push(cur.slice(0, 3));
    trails.push({ path: past, color: [...col, sel ? 255 : dim ? 40 : 150], w: sel ? 3 : 1.25 });
    const fut = { path: [cur.slice(0, 3), ...path.p.filter((q) => q[3] > t).map((q) => q.slice(0, 3))], color: [...col, sel ? 110 : dim ? 25 : held ? 120 : 45], w: sel ? 2 : 1 };
    (held ? falls : futures).push(fut);
    live.push({ pos: cur.slice(0, 3), tr, sel, held });
  }
  const batt = Object.fromEntries(META.batteries.map((b) => [b.id, b]));
  const shots = [], flashes = [], kills = [], nowMs = performance.now();
  for (const f of S.flights) {
    if (t < f.t0) continue;
    const g = Math.min(1, (t - f.t0) / Math.max(1, f.t1 - f.t0)), pts = [];
    for (let k = 0; k <= 24 * g; k++) { const s = k / 24; pts.push([f.slon + (f.lon - f.slon) * s, f.slat + (f.lat - f.slat) * s, f.alt * Math.pow(s, 0.6)]); }
    if (g < 1) shots.push({ path: pts, sel: f.th === selected });
    else {
      const key = `${f.b}${f.th}${f.t1}`;
      if (!flashSeen.has(key)) flashSeen.set(key, nowMs);
      const age = (nowMs - flashSeen.get(key)) / 1000;
      if (age < 1.6) flashes.push({ pos: [f.lon, f.lat, f.alt], age, hit: f.hit });
      if (f.hit && age < 8) kills.push({ pos: [f.lon, f.lat, 0], a: 1 - age / 8 });   // an intercept leaves a mark for a while
    }
  }
  const selTr = S.tracks.find((x) => x.id === selected);
  const leaks = S.tracks.filter((x) => x.status === "leaked").map((x) => PATHS[x.id]?.p.at(-1)).filter(Boolean);
  const batts = S.batteries.map((b) => ({ ...b, ...META.batteries.find((m) => m.id === b.id) }));
  const fb = focus?.k === "b" && batts.find((b) => b.id === focus.id);
  const links = fb ? fb.tasks.map((tid) => live.find((l) => l.tr.id === tid)).filter(Boolean).map((l) => ({ path: [[fb.lon, fb.lat, 0], l.pos] })) : [];
  overlay.setProps({ layers: [
    new deck.ScatterplotLayer({ id: "sites", data: META.sites, getPosition: (d) => [d.lon, d.lat], getRadius: (d) => d.r * 1000,
      getFillColor: (d) => (prio.has(d.id) ? [16, 16, 16, 14] : [0, 0, 0, 0]), stroked: true, lineWidthUnits: "pixels",
      getLineWidth: (d) => (prio.has(d.id) ? 1.5 : 0.75), getLineColor: (d) => (prio.has(d.id) ? [...INK, 200] : [197, 197, 199, 255]),
      updateTriggers: { getFillColor: [...prio].join(), getLineColor: [...prio].join(), getLineWidth: [...prio].join() } }),
    new deck.TextLayer({ id: "countries", data: COUNTRIES, getPosition: (d) => [d.lon, d.lat], getText: (d) => glyphs(nm(d)), getSize: 13,
      getColor: [150, 150, 154, 200], ...LABEL, extensions: [], updateTriggers: { getText: LANG } }),
    new deck.TextLayer({ id: "siteNames", data: META.sites.filter((s) => prio.has(s.id) || map.getZoom() > 6.2), getPosition: (d) => [d.lon, d.lat],
      getText: (d) => glyphs(nm(d)), getSize: (d) => (prio.has(d.id) ? 13 : 12), getColor: (d) => (prio.has(d.id) ? [...INK, 230] : [104, 104, 104, 220]),
      getPixelOffset: [0, -16], fontWeight: 500, ...LABEL, extensions: [],   // few sites, no overlap: always labelled
      updateTriggers: { getText: LANG, getSize: [...prio].join(), getColor: [...prio].join() } }),
    new deck.PathLayer({ id: "airtask", data: (S.units || []).filter((u) => u.state === "tasked" && u.goal),
      getPath: (u) => [[u.lon, u.lat], [u.goal[1], u.goal[0]]], getColor: [...INK, 90], widthUnits: "pixels", getWidth: 1 }),
    new deck.IconLayer({ id: "air", data: S.units || [], getPosition: (u) => [u.lon, u.lat], iconAtlas: ATLAS, iconMapping: ICONS, getIcon: () => "tri",
      sizeUnits: "pixels", getSize: (u) => (isFocus("u", u.id) ? 20 : u.state === "ground" || u.state === "rearm" ? 11 : 15), getAngle: (u) => -bearing(u),
      getColor: (u) => (isFocus("u", u.id) ? BLUE : u.state === "tasked" ? INK : [...GREY, dim ? 90 : 230]), pickable: true, onClick: (i) => select("u", i.object.id),
      updateTriggers: { getColor: [(S.units || []).map((u) => u.state).join(), focus?.id, dim], getSize: focus?.id } }),
    new deck.TextLayer({ id: "airNames", data: (S.units || []).filter((u) => (u.state !== "ground" && u.state !== "rearm") || isFocus("u", u.id)),
      getPosition: (u) => [u.lon, u.lat], getText: (u) => glyphs(nm(u)), getSize: 12, getColor: (u) => (isFocus("u", u.id) ? [...BLUE, 255] : [...INK, 200]),
      getPixelOffset: [0, 16], ...LABEL, extensions: [],   // six aircraft at most: always labelled
      updateTriggers: { getText: LANG, getColor: focus?.id } }),
    new deck.PathLayer({ id: "battlinks", data: links, getPath: (d) => d.path, getColor: [...BLUE, 160], widthUnits: "pixels", getWidth: 1.5 }),
    new deck.IconLayer({ id: "battEdge", data: batts, getPosition: (d) => [d.lon, d.lat], iconAtlas: ATLAS, iconMapping: ICONS, getIcon: () => "sq",
      sizeUnits: "pixels", getSize: (d) => (isFocus("b", d.id) ? 17 : 12), pickable: true, onClick: (i) => select("b", i.object.id),
      getColor: (d) => (isFocus("b", d.id) ? BLUE : d.stock === 0 ? [190, 190, 194, 255] : [...INK, dim ? 90 : 230]),
      updateTriggers: { getColor: [batts.map((b) => b.stock === 0).join(), focus?.id, dim], getSize: focus?.id } }),
    new deck.IconLayer({ id: "battFill", data: batts, getPosition: (d) => [d.lon, d.lat], iconAtlas: ATLAS, iconMapping: ICONS, getIcon: () => "sq",
      sizeUnits: "pixels", getSize: (d) => (isFocus("b", d.id) ? 11 : 8),
      getColor: (d) => (isFocus("b", d.id) ? [255, 255, 255] : d.tasks.length || d.busy ? INK : d.stock === 0 ? [236, 236, 238] : d.reload ? [180, 180, 184] : [255, 255, 255]),
      updateTriggers: { getColor: [batts.map((b) => `${b.tasks.length}${b.busy}${b.stock}${!!b.reload}`).join(), focus?.id], getSize: focus?.id } }),
    new deck.TextLayer({ id: "battNames", data: batts.filter((b) => isFocus("b", b.id) || map.getZoom() > 8.5), getPosition: (d) => [d.lon, d.lat],
      getText: (d) => d.id, getSize: 12, getColor: (d) => (isFocus("b", d.id) ? [...BLUE, 255] : [...INK, 170]), getPixelOffset: [10, 0], getTextAnchor: "start",
      ...LABEL, getCollisionPriority: (d) => (isFocus("b", d.id) ? 200 : d.tasks.length ? 50 : 5), updateTriggers: { getColor: focus?.id } }),
    selTr && !DONE.includes(selTr.status) && new deck.ScatterplotLayer({ id: "ipp", data: [selTr], getPosition: (d) => [d.ipp[0], d.ipp[1]],
      getRadius: (d) => Math.max(d.ipp[2], 0.6) * 1000, filled: true, stroked: true, getFillColor: [...BLUE, 22], getLineColor: [...BLUE, 220], lineWidthUnits: "pixels", getLineWidth: 1.5 }),
    new deck.PathLayer({ id: "future", data: futures, getPath: (d) => d.path, getColor: (d) => d.color, widthUnits: "pixels", getWidth: (d) => d.w }),
    new deck.PathLayer({ id: "falls", data: falls, getPath: (d) => d.path, getColor: (d) => d.color, widthUnits: "pixels", getWidth: (d) => d.w,
      getDashArray: [3, 3], dashJustified: true, extensions: [new deck.PathStyleExtension({ dash: true })] }),
    new deck.TextLayer({ id: "fallPts", data: falls, getPosition: (d) => d.path.at(-1), getText: () => "×", getSize: 14, getColor: (d) => d.color,
      fontFamily: "Figtree, Arial, sans-serif", characterSet: "×" }),
    new deck.PathLayer({ id: "trail", data: trails, getPath: (d) => d.path, getColor: (d) => d.color, widthUnits: "pixels", getWidth: (d) => d.w }),
    new deck.PathLayer({ id: "shots", data: shots, getPath: (d) => d.path, getColor: (d) => (d.sel ? [...BLUE, 255] : [...INK, 110]), widthUnits: "pixels", getWidth: (d) => (d.sel ? 2.5 : 1) }),
    new deck.ScatterplotLayer({ id: "heads", data: live, getPosition: (d) => d.pos, radiusUnits: "pixels", pickable: true, onClick: (i) => select("t", i.object.tr.id),
      getRadius: (d) => (d.sel ? 8 : 5), stroked: true, lineWidthUnits: "pixels", getLineWidth: (d) => (d.sel ? 2.5 : 1.5),
      getFillColor: (d) => (d.sel ? BLUE : [...(ENGAGED.includes(d.tr.status) ? INK : [255, 255, 255]), d.held ? 0 : dim ? 70 : 255]),
      getLineColor: (d) => (d.sel ? [255, 255, 255] : [...(d.held ? [150, 150, 154] : INK), dim ? 60 : 225]) }),
    new deck.TextLayer({ id: "kills", data: kills, getPosition: (d) => d.pos, getText: () => "×", getSize: 16, getColor: (d) => [...INK, 200 * d.a],
      fontFamily: "Figtree, Arial, sans-serif", characterSet: "×" }),
    new deck.ScatterplotLayer({ id: "flash", data: flashes, getPosition: (d) => d.pos, radiusUnits: "pixels", getRadius: (d) => 5 + d.age * 22, filled: false,
      stroked: true, lineWidthUnits: "pixels", getLineWidth: 1.5, getLineColor: (d) => [...(d.hit ? INK : GREY), 255 * (1 - d.age / 1.6)] }),
    new deck.ScatterplotLayer({ id: "leaks", data: leaks, getPosition: (d) => [d[0], d[1]], radiusUnits: "pixels", getRadius: 6, getFillColor: [...HIT, 255],
      stroked: true, getLineColor: [...HIT, 70], lineWidthUnits: "pixels", getLineWidth: 7 }),
  ].filter(Boolean) });
  stripNow(t);
  const nowLine = document.getElementById("raidnow");
  if (nowLine && raidX) nowLine.setAttribute("transform", `translate(${raidX(t)},0)`);
  if (raidX && Math.floor(t * 120 / Math.max(S.end, 1)) !== renderRaid.bin) renderRaid();   // the launch histogram fills as the raid goes
}

// ---------------------------------------------------------------- bar
let lastSkip = { v: 0, at: -1e9 };
const SPEEDS = [[1, 0], [5, 1], [20, 2], [60, 3]];
function renderBar() {
  if (!S) return;
  $("#play").textContent = raidOver() ? T().restart : S.running ? T().pause : T().play;
  $("#play").className = `dc-pill dc-pill--sm ${S.running ? "dc-pill--secondary" : "dc-pill--primary"}`;   // one blue: Play only when it is the next thing to do
  $("#next").textContent = T().next; $("#next").style.display = S.drill || S.mode !== "focus" ? "none" : "";
  $("#mode").style.display = S.drill ? "none" : "";
  if (renderBar.mode !== S.mode + LANG) {
    renderBar.mode = S.mode + LANG;
    $("#mode").innerHTML = ["general", "focus"].map((m) => `<button class="dc-tab" data-mode="${m}" aria-selected="${S.mode === m}">${T().modes[m]}</button>`).join("");
  }
  $("#speedBtn").innerHTML = `${T().speed} <span class="num">${S.speed}×</span> <span aria-hidden="true">▾</span>`;
  if (renderBar.key !== S.speed + LANG) {   // the menu only changes with speed or language; the rest below every poll
    renderBar.key = S.speed + LANG;
    $("#speedMenu").innerHTML = SPEEDS.map(([x, d]) => `<button class="dc-dropdown__item" role="menuitemradio" aria-checked="${x === S.speed}" data-x="${x}">
    <span class="dc-dropdown__text"><span class="dc-dropdown__label num">${x}×</span><span class="dc-dropdown__desc">${T().speedDesc[d]}</span></span></button>`).join("");
  }
  const sc = META.scenarios[S.scenario];
  $("#scenBtn").firstChild.textContent = (S.drill ? `${T().drills}: ${T().drill.fam[S.drill.family]}` : (LANG === "he" ? sc.title_he : sc.title).split(":")[0]) + " ";
  if (S.skipped !== lastSkip.v) lastSkip = { v: S.skipped, at: performance.now() };
  const skip = S.running && S.skipped > 0 && performance.now() - lastSkip.at < 2500;
  $("#note").textContent = skip ? T().skipped(fmtT(S.skipped)) : "";
}

// ---------------------------------------------------------------- decision panel
function reason(i, D) {
  const tt = T(), o = Object.fromEntries(i.opts.map(([k, , total, lenses, pk, launch, unit]) => [k, { total, lenses, pk, launch, unit }]));
  const c = o[D.choice]; if (!c) return "";
  const [act, x] = D.choice.split(":"), pk = tt.pk[pkIdx(c.pk)];
  let s = act === "now" ? tt.whyFire(effName(x), pk)
    : act === "air" ? tt.whyAir(unitName(c.unit), mins(Math.max(0, c.launch - i.t)), pk)
    : act === "cheaper" ? tt.whyWait(effName(batEff(c.unit)), mins(Math.max(0, c.launch - i.t)), pk)
    : act === "wait" ? tt.whyTrack : D.open_ground ? tt.whyHoldOpen : tt.whyHold;
  const runner = Object.keys(o).filter((k) => k !== D.choice).sort((a, b) => o[a].total - o[b].total)[0];
  if (runner) {   // why not the runner-up: the lens where it loses most
    const r = o[runner], d = Object.fromEntries(["local", "raid", "campaign"].map((k) => [k, (r.lenses[k] || 0) - (c.lenses[k] || 0)]));
    const k = Object.keys(d).sort((a, b) => d[b] - d[a])[0];
    if (d[k] > 0.0005) s += " " + (k === "local" ? tt.vs.local(actLabel(runner), money(d[k])) : tt.vs[k](actLabel(runner)));
  }
  return s;
}

function localLine(D) {   // the fire unit's call, in one line: who, how many, and the fact that decided it
  const L = D.local, tt = T().local;
  if (!L || D.status !== "assigned" && D.status !== "inflight") return "";
  const f = L.facts || {}, who = D.assign?.kind === "air" ? unitName(L.unit) : L.unit;
  const why = L.fired ? (f.last ? tt.last : f.look ? tt.look : tt.nolook)
    : f.channels_free < 1 ? tt.busy : f.ready < 1 ? tt.reload : f.friendly ? tt.friendly : f.debris ? tt.debris(siteName(f.debris)) : "";
  const left = D.assign?.kind === "air" ? null : S.batteries.find((b) => b.id === L.unit)?.stock;
  return `<p class="local"><b>${esc(L.fired ? tt.fires(who, L.fired) : tt.holds(who))}</b>${why ? " · " + esc(why) : ""}${L.fired && left != null ? " · " + esc(tt.left(left)) : ""}</p>`;
}

// ---------------------------------------------------------------- key moments: the caption card and the raid timeline
const barsNode = document.createElement("div");   // the live bars keep their DOM across re-renders, so widths animate
barsNode.className = "opts live";
function liveBars(dist, chosen) {   // update the live bars in place: a row per option, sorted, the call highlighted
  if (!dist) { barsNode.innerHTML = ""; return; }
  const keys = Object.keys(dist).sort((a, b) => dist[b] - dist[a]);
  if (barsNode.dataset.keys !== keys.join() || barsNode.dataset.lang !== LANG) {
    barsNode.dataset.keys = keys.join(); barsNode.dataset.lang = LANG;
    barsNode.innerHTML = `<p class="kicker">${T().options}</p>` + keys.map((k) => `<div class="opt" data-o="${esc(k)}"><span class="opt-l">${esc(actLabel(k))}</span>
      <span class="opt-bar"><i></i></span><span class="opt-p num"></span></div>`).join("");
  }
  for (const row of barsNode.querySelectorAll("[data-o]")) {
    const v = dist[row.dataset.o];
    row.classList.toggle("on", row.dataset.o === chosen);
    row.querySelector("i").style.width = `${Math.max(1, Math.round(v * 100))}%`;
    row.querySelector(".opt-p").textContent = `${v < 0.005 ? "<1" : Math.round(v * 100)}%`;
  }
}
function progression(D) {   // the threat's story so far: each call as it was made and firmed up, each shot, the outcome
  const tt = T(), rows = [];
  for (const e of D.log || []) {
    if (e.fire) { rows.push({ t: e.t, html: `<b>${esc(tt.local.fires(unitName(e.unit), e.fire))}</b>` }); continue; }
    const prev = rows.at(-1);
    if (prev?.key === e.key) { prev.p1 = e.p; continue; }
    rows.push({ t: e.t, key: e.key, p0: e.p, p1: e.p, opts: e.opts });
  }
  for (const sh of D.shots || []) if (D.t >= sh.t1) rows.push({ t: sh.t1, html: esc(sh.hit ? tt.shotHit(effName(sh.eff)) : tt.shotMiss(effName(sh.eff))), hit: sh.hit, miss: !sh.hit });
  rows.sort((a, b) => a.t - b.t);
  if (!rows.length) return "";
  const pct = (v) => (v == null ? "" : `${Math.round(v * 100)}%`);
  return `<div class="prog"><p class="kicker">${tt.prog}</p>${rows.map((r, n) => {
    const t = `<span class="t num">−${fmtT(Math.max(0, D.impact - r.t))}</span>`;
    if (!r.key) return `<div class="step${r.hit ? " ok" : r.miss ? " miss" : ""}${n === rows.length - 1 ? " new" : ""}">${t}<span class="c">${r.html}</span><span class="p"></span></div>`;
    const why = r.opts ? reason({ t: r.t, opts: r.opts }, { choice: r.key, open_ground: D.open_ground }).split(". ")[0].replace(/\.$/, "") : "";
    const firm = r.p1 != null && r.p0 != null && Math.abs(r.p1 - r.p0) >= 0.05 ? `${pct(r.p0)} → ${pct(r.p1)}` : pct(r.p1);
    return `<div class="step call${n === rows.length - 1 ? " new" : ""}">${t}<span class="c"><b>${esc(actLabel(r.key))}</b>${why ? `<i>${esc(why)}</i>` : ""}</span><span class="p num">${firm}</span></div>`;
  }).join("")}</div>`;
}
function optionBars(dist, chosen) {   // every option Typical weighed, its probability as a bar, the call highlighted
  if (!dist) return "";
  const keys = Object.keys(dist).sort((a, b) => dist[b] - dist[a]);
  return `<div class="opts"><p class="kicker">${T().options}</p>${keys.map((k) => `<div class="opt${k === chosen ? " on" : ""}">
    <span class="opt-l">${esc(actLabel(k))}</span><span class="opt-bar"><i style="width:${Math.max(1, Math.round(dist[k] * 100))}%"></i></span>
    <span class="opt-p num">${dist[k] < 0.005 ? "<1" : Math.round(dist[k] * 100)}%</span></div>`).join("")}</div>`;
}
function captionBody(m) {   // the moment in one or two plain sentences
  const tt = T(), name = nm(META.threat_types[m.kind]).replace(/ (MRBM|SRBM)$/i, "");
  if (m.type === "letfall") return `<p class="why">${esc(tt.letfall(m.count))}</p>`;
  if (m.type === "hit") {
    const tried = {}; for (const [e] of m.shots || []) tried[e] = (tried[e] || 0) + 1;
    const lines = Object.entries(tried).map(([e, n]) => tt.shotsLine(effName(e), n));
    return `<p class="why">${esc(tt.hitLine(name, siteName(m.site)))} ${esc(lines.length ? lines.join(" ") : tt.noShot)}</p>`;
  }
  if (m.type === "holdfire") return `<p class="why">${esc(m.why === "friendly" ? tt.holdfire.friendly(m.unit) : tt.holdfire.debris(m.unit, siteName(m.debris)))}</p>`;
  if (m.type === "payoff") return `<p class="why">${esc(tt.payoffLine(effName(m.key.split(":")[1])))}</p>`;
  return m.info.opts.length ? `<p class="why">${esc(reason(m.info, { choice: m.key, open_ground: m.open }))}</p>` : "";
}
let raidX = null;
function renderRaid() {   // the whole raid on one line: launches as a histogram (the raid's shape), a labelled marker per key moment
  const el = $("#raid"); if (!S || !META) return;
  const W = Math.max(el.clientWidth, 400), L = 92, R = 12, end = Math.max(S.end, 1), x = (t) => L + (Math.min(Math.max(t, 0), end) / end) * (W - L - R);
  const ms = S.moments || [], tt = T(), NB = 120, bin = Math.floor(tNow * NB / end);
  const key = `${W}|${end}|${ms.length}|${S.beat?.id}|${LANG}|${bin}|${PATH_SCEN}`;
  raidX = x; renderRaid.bin = bin;
  if (key === renderRaid.last) return;
  renderRaid.last = key;
  const counts = new Array(NB).fill(0);
  for (const p of Object.values(PATHS)) counts[Math.min(NB - 1, Math.floor(p.t0 * NB / end))]++;
  const top = Math.max(1, ...counts), bw = (W - L - R) / NB;
  const bars = counts.map((c, i) => c ? `<rect x="${(L + i * bw).toFixed(1)}" y="${(34 - 12 * c / top).toFixed(1)}" width="${Math.max(1, bw - 1).toFixed(1)}" height="${(12 * c / top).toFixed(1)}" fill="${i <= bin ? "#101010" : "#d9d9dc"}" opacity="${i <= bin ? 0.55 : 1}"/>` : "").join("");
  let lastX = -1e9;
  const marks = ms.map((m) => {
    const mx = x(m.t), show = mx - lastX > 58; if (show) lastX = mx;
    return `<g data-m="${m.id}" class="mk${S.beat?.id === m.id ? " on" : ""}${m.type === "hit" ? " hit" : ""}" transform="translate(${mx},34)">
      <circle r="10" class="hitbox"/><circle r="4.5"/>${show ? `<text y="-10" text-anchor="middle">${esc(tt.markShort[m.type])}</text>` : ""}
      <title>${esc(tt.mom[m.type])} · ${esc(nm(META.threat_types[m.kind]))}</title></g>`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} 42" width="${W}" height="42" direction="ltr">
    <text x="4" y="38" class="cap">${tt.raidLine}</text>${bars}
    <line x1="${L}" x2="${W - R}" y1="34" y2="34" stroke="#c9c9cd" stroke-width="1.5"/>${marks}
    <g id="raidnow"><line x1="0" x2="0" y1="14" y2="40" stroke="#101010" stroke-width="1.5"/></g></svg>`;
}

// ---------------------------------------------------------------- the list: every object, its status, one expanded
function trackStatus(tr) {   // -> [text, tone]: what is happening to this threat, in one operative phrase
  const st = T().st;
  if (tr.status === "killed") return [T().status.killed, "ink"];
  if (tr.status === "leaked") return [st.hit(siteName(tr.site)), "hit"];
  if (tr.status === "fell_open") return [T().status.fell_open, "grey"];
  if (tr.status === "inflight") return [st.fired(tr.unit, tr.fired || 1), "ink"];
  if (tr.status === "assigned") return [META.effectors[tr.eff]?.air ? st.air(unitName(tr.unit)) : st.ground(effName(tr.eff), tr.unit), "ink"];
  if (tr.status === "held") return [st.hold, "grey"];
  if (tr.status === "waiting") return [tr.wait_eff ? st.cheaper(effName(tr.wait_eff)) : st.wait, "sub"];
  return [tr.choice ? actLabel(tr.choice) : st.out, "grey"];
}
function battStatus(b) {
  const t = T().bat;
  return b.tasks.length || b.busy ? [t.engaging(Math.max(b.tasks.length, b.busy)), "ink"] : b.stock === 0 ? [t.empty, "grey"] : b.reload ? [t.reloading, "sub"] : [t.ready, "sub"];
}
const weaponsLine = (u) => Object.entries(u.weapons).map(([w, n]) => `${effName(w)} ${n}`).join(" · ");
const threatName = (tid) => { const tr = S.tracks.find((x) => x.id === tid); return tr ? nm(META.threat_types[tr.kind]).replace(/ (MRBM|SRBM)$/i, "") : tid; };

function row(k, id, name, sub, [status, tone], eta) {
  const open = isFocus(k, id);
  return `<button class="obj${open ? " open" : ""}" data-k="${k}" data-id="${esc(id)}" aria-expanded="${open}">
    <span class="obj-main"><span class="obj-name">${esc(name)}</span><span class="obj-sub">${esc(sub)}</span></span>
    <span class="obj-st tone-${tone}">${esc(status)}</span>${eta == null ? "" : `<span class="obj-eta num eta" data-impact="${eta}"></span>`}</button>`;
}

function renderStory() {   // Focus mode: one object, its moment and its call -- the panel is the caption
  const el = $("#list"), tt = T();
  let h = "";
  if (raidOver()) h = recap();
  else if (focus?.k === "t") {
    const tr = S.tracks.find((x) => x.id === focus.id), m = caption && caption.tid === focus.id ? caption : null;
    const src = tr || m;
    if (src) {
      const name = nm(META.threat_types[src.kind]).replace(/ (MRBM|SRBM)$/i, ""), site = tr ? tr.target : m.site, open = tr ? tr.open_ground : m.open;
      const title = m?.type === "letfall" ? esc(tt.salvoOf(tt.clsPl[m.cls], tt.origin[PATHS[m.tid]?.origin] || "")).replace(/^./, (c) => c.toUpperCase()) : esc(name);
      const dest = m?.type === "letfall" ? "" : site ? (open ? `${tt.openGround} ${esc(siteName(site))}` : `${tt.to} ${esc(siteName(site))}`) : "";
      const live = tr && !DONE.includes(tr.status);
      const kick = m && tt.mom[m.type] !== actLabel(m.key) ? tt.mom[m.type] : tt.cls[src.cls];   // never the call twice
      const past = m && S.beat?.id !== m.id;
      h = `<div class="${past ? "past" : ""}"><p class="kicker">${kick}</p>
        <div class="row"><div><h2 class="threat">${title}</h2><div class="target">${dest}</div></div>${live && m?.type !== "letfall" ? `<div class="etabox"><span class="kicker">${tt.impactIn}</span><div class="eta num" data-impact="${tr.impact}"></div></div>` : ""}</div>`;
      const D = detail?.id === focus.id ? detail : null;
      if (m?.type === "letfall" || (m && !D)) {
        const call = m.type === "letfall" || m.type === "hit" || m.type === "holdfire" ? "" : `<div class="what">${esc(actLabel(m.key))}</div>`;
        h += `<div class="story">${call}${captionBody(m)}</div></div>`;
      } else if (D) {   // the story so far, live: the call now, why, the unit's call, the options, and how it got here
        const done = DONE.includes(D.status), i = D.info;
        const head = done ? `<div class="what ${D.status === "leaked" ? "hit" : "ink"}">${tt.status[D.status]}</div>`
          : `<div class="what">${esc(actLabel(D.choice))}</div>${i ? `<p class="why">${esc(reason(i, D))}</p>` : ""}${localLine(D)}`;
        const extra = m && (m.type === "holdfire" || m.type === "hit") ? captionBody(m) : "";
        h += `<div class="story">${head}${extra}${done ? "" : '<div id="bars"></div>'}${progression(D)}</div></div>`;
        liveBars(done ? null : i?.dist, D.choice);
      } else {
        h += threatDetail(true).replace('<div class="obj-detail">', '<div class="story">');
        if (detail?.id === focus.id && detail.info?.dist && !DONE.includes(detail.status)) h += optionBars(detail.info.dist, detail.choice);
        h += "</div>";
      }
    }
  } else if (focus?.k === "b") {
    const b = S.batteries.find((x) => x.id === focus.id);
    if (b) h = `<p class="kicker">${esc(effName(b.eff))}</p><h2 class="threat">${esc(b.id)}</h2>${battDetail(b)}`;
  } else if (focus?.k === "u") {
    const u = S.units.find((x) => x.id === focus.id);
    if (u) h = `<p class="kicker">${esc(T().act.air[u.kind])}</p><h2 class="threat">${esc(nm(u))}</h2>${unitDetail(u)}`;
  }
  if (S.drill) h += drillCard();
  else if (!h) h = (S.moments || []).length || S.running ? "" : `<p class="kicker">${esc(scenTitle())}</p><h2 class="intro">${tt.intro}</h2><p class="why">${tt.pressPlay}</p>`;
  if (!S.drill && !raidOver() && S.running && !S.beat) h += `<p class="seek">${tt.seeking}…</p>`;
  if (!S.drill) h += `<p class="showall"><button class="linkish" data-mode="general">${tt.showAll}</button></p>`;
  el.classList.add("story-mode");
  if (h !== renderStory.last) {
    el.innerHTML = h; renderStory.last = h; stripNow(tNow);
    el.querySelector("#bars")?.replaceWith(barsNode);   // the same bar elements every time: their widths animate
    el.querySelector(".prog .step.new")?.scrollIntoView({ block: "nearest" });
  }
  renderList.last = null;
}

const scenTitle = () => { const sc = META.scenarios[S.scenario]; return sc ? (LANG === "he" ? sc.title_he : sc.title) : ""; };
function recap() {   // the raid is over: tonight's figures and every moment, each one replayable
  const tt = T();
  return `<p class="kicker">${tt.raidOver}</p><h2 class="threat">${esc(scenTitle())}</h2>
    <p class="dline">${tt.recap}</p>${(S.moments || []).map((m) => `<button class="obj" data-m="${m.id}">
      <span class="obj-main"><span class="obj-name">${esc(tt.mom[m.type])}</span><span class="obj-sub">${esc(nm(META.threat_types[m.kind]))}${m.site ? " · " + esc(siteName(m.site)) : ""}</span></span>
      <span class="obj-st tone-${m.type === "hit" ? "hit" : "sub"}">${m.type === "letfall" ? m.count : m.key && m.type !== "hit" && m.type !== "holdfire" ? esc(actLabel(m.key)) : ""}</span></button>`).join("")}`;
}

function renderList() {
  const el = $("#list"); if (!S || !META) return;
  $("#tonight").style.display = S.drill ? "none" : "";
  if (S.mode === "focus" || S.drill) return renderStory();
  renderStory.last = null;
  el.classList.remove("story-mode");
  const tt = T(), top = el.scrollTop;
  const tracks = [...S.tracks].sort((a, b) => a.impact - b.impact);   // stable: a row never jumps when its status changes
  const sec = (k, n, body) => `<section class="sec"><button class="sec-h" data-sec="${k}" aria-expanded="${!collapsed[k]}"><span>${tt.sec[k]}</span><span class="num">${n}</span></button>${collapsed[k] ? "" : body}</section>`;
  const tRow = (tr) => {
    const ty = META.threat_types[tr.kind], done = DONE.includes(tr.status);
    const dest = !tr.target ? tt.cls[tr.cls] : `${tt.cls[tr.cls]} · ${tr.open_ground ? `${tt.openGround} ${siteName(tr.target)}` : siteName(tr.target)}`;
    return row("t", tr.id, nm(ty).replace(/ (MRBM|SRBM)$/i, ""), dest, trackStatus(tr), done ? null : tr.impact) + (isFocus("t", tr.id) ? threatDetail() : "");
  };
  const groups = new Map();   // a salvo: same class, same origin, launched within 3 minutes
  for (const tr of tracks) groups.set(gkey(tr), [...(groups.get(gkey(tr)) || []), tr]);
  const tRows = [...groups.entries()].map(([g, members]) => {
    if (members.length < 3) return members.map(tRow).join("");
    const n = (f) => members.filter(f).length, live = members.filter((x) => !DONE.includes(x.status));
    const tally = [[tt.tally.fall, n((x) => x.status === "held" || x.status === "fell_open")], [tt.tally.engaged, n((x) => ENGAGED.includes(x.status))],
      [tt.tally.killed, n((x) => x.status === "killed")]].filter(([, k]) => k).map(([f, k]) => f(k)).join(" · ");
    const hits = n((x) => x.status === "leaked"), open = openGroups.has(g) || members.some((x) => isFocus("t", x.id));
    const eta = live.length ? Math.min(...live.map((x) => x.impact)) : null;
    return `<button class="obj grp${open ? " gopen" : ""}" data-g="${esc(g)}" aria-expanded="${open}">
      <span class="obj-main"><span class="obj-name">${esc(tt.salvo(members.length, tt.clsPl[members[0].cls], tt.origin[PATHS[members[0].id]?.origin] || ""))}</span><span class="obj-sub">${esc(tally)}</span></span>
      <span class="obj-st tone-${hits ? "hit" : "sub"}">${hits ? esc(tt.tally.hits(hits)) : ""}</span>${eta == null ? "" : `<span class="obj-eta num eta" data-impact="${eta}"></span>`}</button>`
      + (open ? `<div class="grp-body">${members.map(tRow).join("")}</div>` : "");
  }).join("");
  const uRows = S.units.map((u) => row("u", u.id, nm(u), `${T().act.air[u.kind]} · ${weaponsLine(u)}`,
    [u.state === "tasked" && u.task ? `${tt.unit.tasked} ${threatName(u.task)}` : tt.unit[u.state], u.state === "tasked" ? "ink" : "sub"]) + (isFocus("u", u.id) ? unitDetail(u) : "")).join("");
  const bRows = S.batteries.map((b) => row("b", b.id, b.id, `${effName(b.eff)} · ${b.stock == null ? tt.bat.unlimited : tt.bat.sub(b.ready, b.stock)}`, battStatus(b)) + (isFocus("b", b.id) ? battDetail(b) : "")).join("");
  const live = tracks.filter((x) => !DONE.includes(x.status)).length;
  const html = (tracks.length ? sec("t", tt.inAir(live, S.total), tRows) : `<p class="empty">${tt.empty}</p>`) + sec("u", S.units.length, uRows) + sec("b", S.batteries.length, bRows);
  if (html === renderList.last) return;   // re-render only on change: countdowns tick per frame (stripNow), clicks survive
  renderList.last = html;
  el.innerHTML = html;
  el.scrollTop = top;
  stripNow(tNow);
  if (scrollToFocus) { el.querySelector(".obj.open")?.scrollIntoView({ block: "nearest" }); scrollToFocus = false; }
}

const facts = (pairs) => `<dl class="facts">${pairs.map(([k, v]) => `<dt>${esc(k)}</dt><dd class="num">${v}</dd>`).join("")}</dl>`;
const link = (tid) => `<button class="linkish" data-k="t" data-id="${esc(tid)}">${esc(threatName(tid))}</button>`;

function battDetail(b) {
  const t = T().bat;
  return `<div class="obj-detail">${facts([
    [t.onLaunchers, b.ready == null ? t.unlimited : `${b.ready} / ${b.cap ?? "—"}`],
    [t.left, b.stock == null ? t.unlimited : `${b.stock} / ${b.init}`],
    [t.channels, `${b.busy} / ${b.channels}`],
    [t.reload, b.reload ? fmtT(b.reload) : t.none],
    [t.firedTonight, b.fired],
    [t.targets, b.tasks.length ? b.tasks.map(link).join(", ") : t.none]])}</div>`;
}
function unitDetail(u) {
  const t = T().unit;
  return `<div class="obj-detail">${facts([[t.state, esc(t[u.state])], [t.weapons, esc(weaponsLine(u))], [t.task, u.task ? link(u.task) : T().bat.none]])}</div>`;
}

function threatDetail(story = false) {   // the call on the expanded threat: what, why, the fire unit's line, then the detail disclosure
  const tt = T(), D = detail;
  if (caption && caption.tid === focus.id && !pinned) return "";   // the caption card is explaining this one
  if (!D || D.id !== focus.id) return `<div class="obj-detail"><p class="empty">${tt.deciding}</p></div>`;
  const done = DONE.includes(D.status), i = D.info;
  let h = pinned && S.mode === "focus" && !S.drill ? `<p class="kicker"><button id="followBtn" class="linkish">${tt.backStory}</button></p>` : "";
  if (done) return `<div class="obj-detail">${h}<div class="what" style="color:${D.status === "leaked" ? "var(--hit)" : "var(--dc-ink-2)"}">${tt.status[D.status]}</div></div>`;
  if (!i) return `<div class="obj-detail">${h}<p class="empty">${tt.notYet}</p></div>`;
  h += `<div class="what">${esc(actLabel(D.choice))}</div><p class="why">${esc(reason(i, D))}</p>${localLine(D)}`;
  const opts = i.dist ? Object.keys(i.dist).sort((a, b) => i.dist[b] - i.dist[a]) : [];
  h += `<details class="detail" ${detailOpen ? "open" : ""} ontoggle="detailOpen = this.open"><summary>${tt.details}</summary>
    ${opts.length && !story ? `<p class="dline">${tt.options}</p>${opts.map((k) => `<div class="alt${k === D.choice ? " chosen" : ""}"><span>${esc(actLabel(k))}</span><span class="num">${Math.round(i.dist[k] * 100)}%</span></div>`).join("")}` : ""}
    <p class="dline">${tt.reads}</p><pre>${esc(i.state)}</pre></details>`;
  return `<div class="obj-detail">${h}</div>`;
}
let detailOpen = false;

// ---------------------------------------------------------------- tonight
function drillCard() {   // the curated case: every option played out to the end, the best marked, Typical's pick marked
  const D = S.drill, tt = T(), d = tt.drill, tol = Math.max(0.05, 0.02 * D.costs[0][1]), lo = D.costs[0][1], hi = Math.max(...D.costs.map((c) => c[1]), 1e-9);
  return `<div class="drill"><p class="kicker">${d.kicker(D.case + 1, D.cases)}</p><p class="dline">${d.played}</p>
    ${D.costs.map(([k, c]) => `<div class="opt${D.typical === k ? " on" : ""}"><span class="opt-l">${esc(actLabel(k))}${c - lo <= tol ? ` · ${d.best}` : ""}</span>
      <span class="opt-bar"><i style="width:${Math.max(1, Math.round(100 * c / hi))}%"></i></span><span class="opt-p num">${money(c)}</span></div>`).join("")}
    ${D.typical ? "" : `<p class="dline">${d.thinking}</p>`}
    ${D.cases > 1 ? `<button id="nextCase" class="dc-pill dc-pill--secondary dc-pill--sm">${d.next}</button>` : ""}</div>`;
}

function renderTonight() {
  if (!S || S.drill) return;
  const L = S.ledger, ty = L.typical, tt = T();
  if (!ty) return;
  $("#tonight").innerHTML = `<p class="kicker">${tt.tonight}</p><div class="stats three">
      <div class="stat"><div class="v num">${money(ty.spent)}</div><div class="l">${tt.spent}</div></div>
      <div class="stat"><div class="v num">${ty.kills}</div><div class="l">${tt.intercepted}</div></div>
      <div class="stat"><div class="v num ${ty.leaks ? "hit" : ""}">${ty.leaks}</div><div class="l">${tt.hits}</div></div>
    </div>${S.stats && (S.stats.fall || S.stats.cheaper) ? `<p class="held">${esc(tt.held(S.stats.fall, S.stats.cheaper, "\u2066" + money(S.stats.kept) + "\u2069"))}</p>` : ""}
    <div class="mags">${magRows(ty)}</div>`;
}

const MAG_ORDER = ["arrow3", "arrow2", "thaad", "sm3", "dsling", "sm6", "ironDome", "ironBeam", "ew"];
const MAG_ALWAYS = ["arrow3", "arrow2", "dsling", "ironDome"];
let firedAt = {};   // interceptor -> [rounds fired tonight, when that last changed]: a row lights up as it fires
function magRows(ty) {   // every interceptor in play: what is left, what tonight has used; unlimited ones (laser, EW) as a count
  const fired = ty.rounds || {}, now = performance.now(), tt = T();
  return MAG_ORDER.filter((e) => fired[e] || (MAG_ALWAYS.includes(e) && ty.init?.[e])).map((e) => {
    const n = fired[e] || 0;
    if (!firedAt[e] || firedAt[e][0] !== n) firedAt[e] = [n, firedAt[e] ? now : -1e9];   // first sight is not a shot
    const init = ty.init?.[e], left = ty.left?.[e];
    const stock = init ? `<span class="opt-bar"><i style="width:${Math.round(100 * left / init)}%"></i></span><span class="num">${left}/${init}</span>`
      : `<span class="opt-bar"></span><span class="num">∞</span>`;
    return `<div class="mag${now - firedAt[e][1] < 1500 ? " fresh" : ""}"><span>${esc(effName(e))}</span>${stock}<span class="num d">${n ? "−" + n : ""}</span></div>`;
  }).join("");
}

// ---------------------------------------------------------------- flight strip (the threat's story over time)
let stripGeom = null;
function renderStrip() {   // one lifeline per threat: launch -> impact, the calls on it, shots as ticks, now as a blue dot
  const el = $("#life"), tt = T();
  if (!detail || !META) { el.innerHTML = ""; stripGeom = null; return; }
  const D = detail, W = Math.max(el.clientWidth, 400), Y = 46, PAD = 92;
  const X = (t) => PAD + ((Math.min(Math.max(t, D.t0), D.impact) - D.t0) / Math.max(D.impact - D.t0, 1)) * (W - PAD - 80);
  let g = `<line x1="${X(D.t0)}" x2="${X(D.impact)}" y1="${Y}" y2="${Y}" stroke="#e2e2e4" stroke-width="4" stroke-linecap="round"/>
    <line id="pastline" x1="${X(D.t0)}" x2="${X(D.t0)}" y1="${Y}" y2="${Y}" stroke="#101010" stroke-width="4" stroke-linecap="round"/>`;
  const seen = new Set(), calls = [];   // each call once (A, B, A, B -> A, B), the final call always
  D.history.forEach(([t, key], n) => { if (!seen.has(key) || n === D.history.length - 1) { seen.add(key); calls.push([t, key, n === D.history.length - 1]); } });
  let lastEnd = -1e9;
  for (let n = calls.length - 1, right = 1e9; n >= 0; n--) {   // from the final call back: drop a label that would collide
    const [t, key, last] = calls[n], x = X(t), w = 8 * actLabel(key).length + 10;
    if (!last && x + w > right) continue;
    right = x;
    const col = last ? "#2548f6" : "#5d5d5d";
    g += `<line x1="${x}" x2="${x}" y1="${Y - 12}" y2="${Y}" stroke="${col}" stroke-width="1"/><text x="${x}" y="${Y - 16}" style="fill:${col}${last ? ";font-weight:500" : ""}">${esc(actLabel(key))}</text>`;
    lastEnd = Math.max(lastEnd, x + w);
  }
  for (const s of D.shots) {   // shots below the line
    const fin = D.t >= s.t1;
    g += `<text x="${X(fin ? s.t1 : s.t0)}" y="${Y + 24}" text-anchor="middle" style="fill:${fin ? (s.hit ? "#101010" : "#686868") : "#2548f6"}">${fin ? (s.hit ? "✓" : "✕") : "↑"} ${esc(effName(s.eff))}</text>`;
  }
  g += `<text x="${X(D.impact) + 10}" y="${Y + 5}" style="fill:#101010">${tt.impact[0].toUpperCase() + tt.impact.slice(1)}</text>`;
  g += `<text x="4" y="${Y + 5}" class="cap">${tt.threatLine}</text>`;
  g += `<circle id="nowdot" cx="0" cy="${Y}" r="6" fill="#2548f6" stroke="#fff" stroke-width="2"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} 80" width="${W}" height="80" direction="ltr">${g}</svg>`;
  stripGeom = { X, t0: D.t0 };
  stripNow(tNow);
}
function stripNow(t) {   // the now dot, the flown part and the countdown move every frame, between re-renders
  const d = document.getElementById("nowdot"), p = document.getElementById("pastline");
  if (d && stripGeom) { d.setAttribute("cx", stripGeom.X(t)); p.setAttribute("x2", stripGeom.X(t)); }
  for (const e of document.querySelectorAll(".eta[data-impact]")) e.textContent = fmtT(Math.max(0, +e.dataset.impact - t));
}

// ---------------------------------------------------------------- settings sheet
function buildSheet() {
  if (!S) return;
  const tt = T(), c = S.cfg;
  const rng = (attrs, label, val, shown) => `<label class="rng"><span>${label}</span><input type="range" ${attrs} value="${val}"><output class="num">${shown}</output></label>`;
  const heCls = { B: "טיל בליסטי", R: "רקטה", C: "טיל שיוט", D: "כטב״ם" }, enCls = { B: "Ballistic missile", R: "Rocket", C: "Cruise missile", D: "Drone" };
  const whLabel = (k) => (META.threat_types[k] ? nm(META.threat_types[k]) : (LANG === "he" ? heCls : enCls)[k]);
  $("#sheet").innerHTML = `
    <div class="field"><p class="kicker">${tt.posture}</p><div class="dc-tabs dc-tabs--sm" id="reserve">${tt.postures.map((p, k) => `<button class="dc-tab" data-r="${k}" aria-selected="${c.reserve === k}">${p}</button>`).join("")}</div></div>
    <div class="field"><p class="kicker">${tt.prio}</p><div class="chips" id="prio">${META.sites.map((s) => `<button class="chip" data-site="${s.id}" aria-pressed="${c.priority_sites.includes(s.id)}">${esc(nm(s))}</button>`).join("")}</div></div>
    <div class="field"><p class="kicker">${tt.assets}</p><div class="chips"><button class="chip" id="allied" aria-pressed="${c.allied}">${tt.allied}</button><button class="chip" id="weather" aria-pressed="${c.iron_beam_weather === "cloudy"}">${tt.weather}</button></div></div>
    <div class="field"><p class="kicker">${tt.knobs}</p>
      ${rng('id="pk" min="0.6" max="1.1" step="0.05"', tt.pkScale, c.pk_scale, c.pk_scale)}
      ${rng('id="warn" min="0" max="30" step="1"', tt.warn, Math.round(c.forecast_s / 60), Math.round(c.forecast_s / 60))}</div>
    <div class="field"><p class="kicker">${tt.dmg}</p>
      ${Object.keys(c.tier_value).map((k) => rng(`data-tier="${k}" min="0" max="5000" step="50"`, tt.tier[k], Math.round(c.tier_value[k]), `$${Math.round(c.tier_value[k])}M`)).join("")}
      ${rng('id="pmul" min="1" max="10" step="0.5"', tt.prioMult, c.priority_multiplier, `×${c.priority_multiplier}`)}
      <details class="more"><summary>${tt.warhead}</summary>${Object.keys(c.warhead).map((k) => rng(`data-wh="${k}" min="0" max="2" step="0.01"`, esc(whLabel(k)), c.warhead[k], c.warhead[k])).join("")}</details>
      <details class="more"><summary>${tt.perSite}</summary><div class="sitegrid">${META.sites.map((s) => `<span>${esc(nm(s))}</span><input data-sv="${s.id}" type="number" min="0" step="50" value="${c.site_value[s.id] ?? ""}" placeholder="${Math.round(c.tier_value[s.tier])}" aria-label="${esc(nm(s))}">`).join("")}</div></details></div>
    <p class="note">${tt.note}</p>`;
  const set = (patch) => api("/api/config", patch).then((r) => { S.cfg = r.cfg; });
  $("#reserve").onclick = (e) => { const b = e.target.closest("[data-r]"); if (!b) return; document.querySelectorAll("#reserve .dc-tab").forEach((x) => x.setAttribute("aria-selected", x === b)); set({ reserve: +b.dataset.r }); };
  const press = (b) => { const on = b.getAttribute("aria-pressed") !== "true"; b.setAttribute("aria-pressed", on); return on; };
  $("#prio").onclick = (e) => { const b = e.target.closest("[data-site]"); if (!b) return; press(b); set({ priority_sites: [...document.querySelectorAll('#prio [aria-pressed="true"]')].map((x) => x.dataset.site) }); };
  $("#allied").onclick = (e) => set({ allied: press(e.currentTarget) });
  $("#weather").onclick = (e) => set({ iron_beam_weather: press(e.currentTarget) ? "cloudy" : "clear" });
  const slide = (el, fmt, patch) => (el.onchange = () => { el.nextElementSibling.textContent = fmt(el.value); set(patch()); });
  const all = (sel, key) => Object.fromEntries([...document.querySelectorAll(sel)].map((x) => [x.dataset[key], +x.value]));
  slide($("#pk"), (v) => v, () => ({ pk_scale: +$("#pk").value }));
  slide($("#warn"), (v) => v, () => ({ forecast_s: 60 * +$("#warn").value }));
  slide($("#pmul"), (v) => `×${v}`, () => ({ priority_multiplier: +$("#pmul").value }));
  document.querySelectorAll("[data-tier]").forEach((r) => slide(r, (v) => `$${v}M`, () => ({ tier_value: all("[data-tier]", "tier") })));
  document.querySelectorAll("[data-wh]").forEach((r) => slide(r, (v) => v, () => ({ warhead: all("[data-wh]", "wh") })));
  document.querySelectorAll("[data-sv]").forEach((inp) => (inp.onchange = () => set({ site_value: Object.fromEntries([...document.querySelectorAll("[data-sv]")].filter((x) => x.value !== "").map((x) => [x.dataset.sv, +x.value])) })));
}

window.addEventListener("resize", () => { renderStrip(); renderRaid.last = null; renderRaid(); });
init();
