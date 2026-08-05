"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Bot, LoaderCircle, Send, Sparkles } from "lucide-react";
import { chatWithAI, getApiError } from "@/lib/api";
import type { PredictionResponse } from "@/types";
import { Button } from "@/components/ui/button";

type ChatMessage = { role: "user" | "assistant"; content: string };
const questions = ["Explain this disease", "Can it spread?", "How should I use the recommended herbs?", "How long does recovery take?", "When should I consult a doctor?"];

export default function ChatPage() {
  const [analysis, setAnalysis] = useState<PredictionResponse | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { try { setAnalysis(JSON.parse(sessionStorage.getItem("diagnosis") ?? "null")); } catch { setError("Your saved analysis could not be loaded."); } }, []);
  const ask = async (question: string) => {
    if (!analysis?.prediction || !question.trim() || loading) return;
    const prompt = question.trim(); setMessages((current) => [...current, { role: "user", content: prompt }]); setInput(""); setLoading(true); setError("");
    try { const response = await chatWithAI({ prediction: analysis.prediction.disease, confidence: analysis.prediction.confidence, disease_information: analysis.disease_information ?? {}, herbs: analysis.recommended_herbs ?? [], question: prompt }); setMessages((current) => [...current, { role: "assistant", content: response.answer || "I could not prepare an answer right now." }]); }
    catch (event) { setError(getApiError(event).message); }
    finally { setLoading(false); }
  };
  if (!analysis?.prediction) return <main className="mx-auto flex min-h-[calc(100vh-5.5rem)] max-w-2xl items-center px-6 py-16 text-center"><div className="w-full rounded-[2rem] border border-border/70 bg-card/85 p-10 shadow-[0_24px_70px_rgba(0,0,0,.07)]"><Bot className="mx-auto text-primary" size={30}/><h1 className="mt-5 text-3xl font-semibold">AI chat is ready after analysis.</h1><p className="mt-3 text-lg leading-8 text-muted-foreground">Complete a skin analysis first so the assistant can answer with your prediction and recommendation context.</p><Link href="/diagnose" className="mt-8 inline-flex rounded-xl bg-primary px-5 py-3 text-sm font-medium text-primary-foreground">Start analysis</Link></div></main>;
  return <main className="mx-auto max-w-5xl px-6 py-12"><div className="flex items-start gap-4"><span className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-primary/10 text-primary"><Bot size={23}/></span><div><p className="text-sm font-medium uppercase tracking-[.25em] text-primary">AI healthcare assistant</p><h1 className="mt-2 text-4xl font-semibold">Ask about {analysis.prediction.disease}.</h1><p className="mt-3 text-muted-foreground">This chat is initialized with your completed skin-analysis context.</p></div></div><section className="mt-8 rounded-[2rem] border border-border/70 bg-card/85 shadow-[0_24px_70px_rgba(0,0,0,.07)]"><div className="min-h-80 space-y-4 p-6">{messages.length === 0 && <div className="rounded-2xl bg-muted/60 p-5 text-sm leading-7 text-muted-foreground">Ask a suggested question or write your own. The assistant can explain the prediction and the herbal recommendations, but it does not replace professional medical care.</div>}{messages.map((message,index)=><div key={`${message.role}-${index}`} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}><p className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-7 ${message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted text-foreground"}`}>{message.content}</p></div>)}{loading&&<div className="flex items-center gap-2 text-sm text-muted-foreground"><LoaderCircle className="animate-spin" size={16}/>Thinking…</div>}{error&&<p className="rounded-xl bg-destructive/10 p-3 text-sm text-destructive">{error}</p>}</div><div className="border-t border-border/60 p-5"><div className="flex flex-wrap gap-2">{questions.map((question)=><button key={question} disabled={loading} onClick={()=>void ask(question)} className="rounded-full border border-border px-3 py-2 text-sm text-muted-foreground hover:border-primary hover:text-foreground disabled:opacity-50">{question}</button>)}</div><div className="mt-4 flex gap-2"><input value={input} onChange={(event)=>setInput(event.target.value)} onKeyDown={(event)=>{if(event.key === "Enter") void ask(input);}} placeholder="Ask the healthcare assistant" className="min-w-0 flex-1 rounded-xl border border-border bg-background px-4 py-3 text-sm outline-none focus:border-primary"/><Button disabled={!input.trim()||loading} onClick={()=>void ask(input)}><Send size={16}/>Send</Button></div></div></section></main>;
}
