"use client";
import { Check, Palette } from "lucide-react";
import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";

const themes = [["herbal","Herbal Green","bg-emerald-500"],["dark","Dark","bg-zinc-700"],["light","Light","bg-amber-300"],["midnight","Midnight Blue","bg-blue-700"],["purple","Purple AI","bg-violet-600"]] as const;
export default function ThemeToggle(){const {theme,setTheme}=useTheme();const [mounted,setMounted]=useState(false);useEffect(()=>setMounted(true),[]);return <div className="relative group"><Button variant="outline" size="icon" className="rounded-full" aria-label="Choose color theme"><Palette size={17}/></Button><div className="invisible absolute right-0 top-11 z-50 w-48 rounded-2xl border border-border bg-popover p-1.5 opacity-0 shadow-2xl transition group-focus-within:visible group-focus-within:opacity-100 group-hover:visible group-hover:opacity-100">{themes.map(([id,name,color])=><button key={id} onClick={()=>setTheme(id)} className="flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left text-sm hover:bg-muted"><span className={`h-3 w-3 rounded-full ${color}`}/>{name}{mounted&&theme===id?<Check className="ml-auto h-4 w-4 text-primary"/>:null}</button>)}</div></div>}
