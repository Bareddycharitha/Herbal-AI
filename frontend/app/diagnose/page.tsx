import Link from "next/link";
import { ArrowRight, Leaf, ScanSearch } from "lucide-react";

const services = [
  { href: "/skin-analysis", icon: ScanSearch, label: "Skin Disease Analysis", tone: "from-cyan-500/15 to-blue-600/10 border-cyan-500/25", iconTone: "bg-cyan-500/15 text-cyan-600 dark:text-cyan-400", description: "Upload a close-up image of affected skin for disease detection, Grad-CAM explanation, AI summary, herbal recommendations, and a downloadable report.", action: "Start Skin Analysis" },
  { href: "/herb-identification", icon: Leaf, label: "Medicinal Herb Identification", tone: "from-emerald-500/15 to-green-600/10 border-emerald-500/25", iconTone: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400", description: "Identify a medicinal plant leaf and explore its scientific classification, traditional uses, medicinal properties, and related skin conditions.", action: "Identify Herb" },
];

export default function DiagnosePage() {
  return <main className="mx-auto max-w-[1450px] px-6 py-16 sm:px-10"><div className="mx-auto max-w-3xl text-center"><p className="text-sm font-medium uppercase tracking-[.28em] text-primary">AI services</p><h1 className="mt-3 text-4xl font-semibold sm:text-5xl">Choose what you want to analyze.</h1><p className="mt-5 text-lg leading-8 text-muted-foreground">Two independent AI workflows, designed for different healthcare questions.</p></div><div className="mx-auto mt-12 grid max-w-6xl gap-6 lg:grid-cols-2">{services.map(({href,icon:Icon,label,tone,iconTone,description,action})=><article key={href} className={`rounded-[2rem] border bg-gradient-to-br ${tone} p-7 shadow-[0_24px_70px_rgba(0,0,0,.07)] transition hover:-translate-y-1`}><span className={`grid h-14 w-14 place-items-center rounded-2xl ${iconTone}`}><Icon size={27}/></span><h2 className="mt-7 text-3xl font-semibold">{label}</h2><p className="mt-4 min-h-28 text-base leading-7 text-muted-foreground">{description}</p><Link href={href} className="mt-8 inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-3 text-sm font-medium text-primary-foreground">{action}<ArrowRight size={17}/></Link></article>)}</div></main>;
}
