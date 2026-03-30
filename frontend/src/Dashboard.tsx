import { useState, useEffect, useRef } from 'react';
import { Search, Activity, Database, Play, Download, Loader2, Cpu, Layout, Terminal, Sparkles, ExternalLink, Square, ArrowUp, ArrowDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface MiningStatus {
    is_running: boolean;
    logs: string[];
    current_step: string;
}

interface MiningResult {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [key: string]: any;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8001`;

const SUGGESTIONS = {
    taskIds: ["Biomass_Carbon_Task1", "SPION_Hyperthermia_RunA", "LiBattery_Study_01", "Carbon_EDLC_Model", "EnergyStorage_Task_1"],
    queries: [
        "Biomass-based carbon supercapacitors capacitance",
        "solid state electrolyte lithium ion batteries",
        "magnetic hyperthermia SPION machine learning",
        "biomass carbon EDLC performance prediction"
    ],
    ranges: [10, 20, 30, 50],
    attributes: {
        "Biomass Carbon": "specific capacitance, current density, specific surface area, micropore volume, pore volume, nitrogen content, ID IG ratio, pore size, oxygen content, carbon oxygen ratio, scan rate, potential window, electrolyte type, micropore surface area, carbon content, energy density, power density",
        "SPION": "specific absorption rate, AMF amplitude, AMF frequency, SPION concentration, core diameter, saturation magnetization, core surface area, core volume, particle shape, Mn Fe ratio, Zn Fe ratio, Co Fe ratio, Mg Fe ratio, diameter standard deviation, surface area volume ratio, coercivity, remanence, Ms Mr ratio, temperature, coating presence, coating type, suspension medium",
        "Lithium-ion": "lithium ion conductivity, activation energy, electrochemical stability window, band gap, phase stability, energy above hull, lithium ion diffusivity, interfacial stability, mean square displacement, electronegativity difference, packing efficiency, packing fraction, volume per atom, shear modulus, bulk modulus, oxidation limit, reduction limit, electronic conductivity"
    }
};

const Dashboard = () => {
    const [query, setQuery] = useState('');
    const [attributes, setAttributes] = useState('');
    const [taskName, setTaskName] = useState('');
    const [limit, setLimit] = useState<number>(10);
    const [activeField, setActiveField] = useState<string | null>(null);
    const [status, setStatus] = useState<MiningStatus>({ is_running: false, logs: [], current_step: 'Idle' });
    const [results, setResults] = useState<MiningResult[]>([]);
    const [jobId, setJobId] = useState<string | null>(() => localStorage.getItem('mining_job_id'));
    const [isCopied, setIsCopied] = useState(false);
    const logEndRef = useRef<HTMLDivElement>(null);
    const logContainerRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        logContainerRef.current?.scrollTo({ top: logContainerRef.current.scrollHeight, behavior: "smooth" });
    };

    const scrollToTop = () => {
        logContainerRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    };

    const handleCopyAll = () => {
        if (!status.logs || status.logs.length === 0) return;
        const text = status.logs.join('\n');
        navigator.clipboard.writeText(text).then(() => {
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), 2000);
        });
    };

    useEffect(() => {
        scrollToBottom();
    }, [status.logs]);

    const fetchResults = async (currentJobId: string | null) => {
        if (!currentJobId) return;
        try {
            const res = await fetch(`${API_BASE_URL}/results/${currentJobId}`);
            const data = await res.json();
            setResults(data.data || []);
        } catch (e) {
            console.error("Failed to fetch results", e);
        }
    };

    useEffect(() => {
        if (jobId) {
            localStorage.setItem('mining_job_id', jobId);

            let interval: ReturnType<typeof setInterval> | null = null;

            const poll = async () => {
                try {
                    const res = await fetch(`${API_BASE_URL}/status/${jobId}`);
                    if (res.ok) {
                        const data = await res.json();
                        setStatus(data);
                        // Stop polling if job has reached a terminal state, and always load results
                        const terminal = ["Completed", "Failed", "Stopped", "Idle"];
                        const isDone = !data.is_running && terminal.some(s => data.current_step?.startsWith(s));
                        if (isDone) {
                            if (interval) clearInterval(interval);
                            fetchResults(jobId); // Always load results for a completed session
                        }
                    }
                } catch {
                    setStatus((prev) => ({ ...prev, current_step: "Server Offline" }));
                    if (interval) clearInterval(interval); // Stop retrying on server offline
                }
            };

            // Fetch immediately on mount (handles already-completed sessions from localStorage)
            poll();
            interval = setInterval(poll, 2000);
            return () => { if (interval) clearInterval(interval); };
        } else {
            localStorage.removeItem('mining_job_id');
            // eslint-disable-next-line react-hooks/set-state-in-effect
            setStatus({ is_running: false, logs: [], current_step: 'Idle' });
            setResults([]);
        }
    }, [jobId]);

    const handleStart = async () => {
        if (!query.trim() || !taskName.trim()) {
            alert("Please provide both a Task Identifier and a Search Objective.");
            return;
        }
        // Clear previous logs and results for a fresh start
        setStatus(prev => ({ ...prev, logs: [] }));
        setResults([]);

        try {
            const res = await fetch(`${API_BASE_URL}/start-mining`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    keywords: [], // Backend will auto-extract from query
                    attributes: attributes.split(',').map(s => s.trim()).filter(Boolean),
                    limit,
                    task_name: taskName,
                    job_id: jobId
                })
            });
            const data = await res.json();
            if (data.job_id) {
                setJobId(data.job_id);
            }
        } catch {
            alert("Failed to start pipeline");
        }
    };

    const handleStop = async () => {
        if (!jobId) return;
        try {
            const res = await fetch(`${API_BASE_URL}/stop-mining/${jobId}`, {
                method: 'POST'
            });
            if (res.ok) {
                setStatus((prev) => ({ ...prev, current_step: "Stopping..." }));
                // Automatically clear the session state as requested
                setJobId(null);
                setQuery('');
                setAttributes('');
                setTaskName('');
            } else {
                alert("Failed to stop mining session");
            }
        } catch {
            alert("Failed to communicate with server");
        }
    };

    const handleNewSession = () => {
        if (confirm("Start a new session? This will clear your current view (results remain on server).")) {
            setJobId(null);
            setQuery('');
            setAttributes('');
            setTaskName('');
        }
    };

    return (
        <div className="relative min-h-screen text-foreground">
            {/* Mesh Background */}
            <div className="mesh-bg" />

            {/* Content Container */}
            <main className="max-w-7xl mx-auto px-6 py-12 lg:py-20 space-y-12">

                {/* Header Section */}
                <header className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.6 }}
                    >
                        <div className="flex items-center gap-3 mb-2">
                            <div className="p-2.5 bg-primary/10 rounded-xl border border-primary/20 shadow-[0_0_15px_hsla(38,92%,50%,0.2)]">
                                <Layout className="text-primary w-6 h-6" />
                            </div>
                            <span className="text-xs font-bold tracking-[0.2em] text-primary uppercase">Research Intelligence</span>
                        </div>
                        <h1 className="text-4xl md:text-5xl font-bold tracking-tight bg-gradient-to-r from-foreground via-foreground/90 to-foreground/60 bg-clip-text text-transparent">
                            Research Mining <span className="text-primary tracking-tighter italic">Pro</span>
                        </h1>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center gap-4 glass-card px-6 py-3 rounded-2xl"
                    >
                        <div className="status-glow">
                            <span className={`status-glow-ping ${status.is_running ? 'bg-emerald-500' : 'bg-slate-200'}`}></span>
                            <span className={`status-glow-dot ${status.is_running ? 'bg-emerald-500' : 'bg-slate-200'}`}></span>
                        </div>
                        <div className="text-sm font-medium">
                            <span className="text-label font-bold uppercase tracking-widest text-[10px]">Session:</span>
                            <span className="ml-2 text-primary font-mono text-[12px]">{jobId || 'None'}</span>
                        </div>
                        <div className="text-sm font-medium ml-4">
                            <span className="text-label font-bold uppercase tracking-widest text-[10px]">Node Status:</span>
                            <span className={`ml-2 text-[11px] font-bold ${status.is_running ? 'text-emerald-400' : 'text-label'}`}>
                                {status.is_running ? 'Pipeline Active' : 'System Ready'}
                            </span>
                        </div>
                    </motion.div>
                </header>

                <div className="flex flex-col gap-8">
                    {/* Configuration Panel */}
                    <motion.section
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="glass-card rounded-[2rem] p-8"
                    >
                        <div className="flex items-center gap-2 mb-6">
                            <Search className="w-5 h-5 text-muted" />
                            <h2 className="text-xl font-bold tracking-tight text-foreground">Configuration</h2>
                        </div>

                        <div className="space-y-8">
                            <div className="space-y-2">
                                <label className="text-[10px] uppercase tracking-[0.15em] text-label font-bold ml-1">Task Identifier</label>
                                <input
                                    value={taskName}
                                    onChange={(e) => setTaskName(e.target.value)}
                                    onFocus={() => setActiveField('taskName')}
                                    className="glass-input w-full font-bold text-primary"
                                    placeholder="Enter Task Identifier (e.g. Biomass_Carbon_Task1)"
                                />
                                <AnimatePresence>
                                    {activeField === 'taskName' && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            exit={{ opacity: 0, height: 0 }}
                                            className="overflow-hidden"
                                        >
                                            <div className="flex flex-wrap gap-2 mt-2 ml-1">
                                                <span className="text-[9px] uppercase tracking-wider text-label/60 font-bold self-center mr-1">Suggestions:</span>
                                                {SUGGESTIONS.taskIds.map((id) => (
                                                    <button
                                                        key={id}
                                                        onClick={() => setTaskName(id)}
                                                        className="text-[9px] px-2 py-0.5 rounded-full bg-primary/5 border border-primary/10 text-primary/70 hover:bg-primary/20 hover:text-primary transition-all"
                                                    >
                                                        {id}
                                                    </button>
                                                ))}
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>

                            <div className="space-y-2">
                                <label className="text-[10px] uppercase tracking-[0.15em] text-label font-bold ml-1">Search Objective</label>
                                <input
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    onFocus={() => setActiveField('query')}
                                    className="glass-input w-full"
                                    placeholder="Enter Search Objective (e.g. Biomass-based carbon supercapacitors)"
                                />
                                <AnimatePresence>
                                    {activeField === 'query' && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            exit={{ opacity: 0, height: 0 }}
                                            className="overflow-hidden"
                                        >
                                            <div className="flex flex-wrap gap-2 mt-2 ml-1">
                                                <span className="text-[9px] uppercase tracking-wider text-label/60 font-bold self-center mr-1">Suggestions:</span>
                                                {SUGGESTIONS.queries.map((q) => (
                                                    <button
                                                        key={q}
                                                        onClick={() => setQuery(q)}
                                                        className="text-[9px] px-2 py-0.5 rounded-full bg-primary/5 border border-primary/10 text-primary/70 hover:bg-primary/20 hover:text-primary transition-all text-left"
                                                    >
                                                        {q}
                                                    </button>
                                                ))}
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>

                            <div className="space-y-2">
                                <label className="text-[10px] uppercase tracking-[0.15em] text-label font-bold ml-1">Download Range</label>
                                <input
                                    type="number"
                                    value={limit || ''}
                                    onChange={(e) => {
                                        const val = parseInt(e.target.value);
                                        setLimit(isNaN(val) ? 0 : val);
                                    }}
                                    onFocus={() => setActiveField('limit')}
                                    onWheel={(e) => (e.target as HTMLInputElement).blur()}
                                    className="glass-input w-full font-mono text-primary"
                                    min="1"
                                    max="1000"
                                />
                                <AnimatePresence>
                                    {activeField === 'limit' && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            exit={{ opacity: 0, height: 0 }}
                                            className="overflow-hidden"
                                        >
                                            <div className="flex flex-wrap gap-2 mt-2 ml-1">
                                                <span className="text-[9px] uppercase tracking-wider text-label/60 font-bold self-center mr-1">Suggestions:</span>
                                                {SUGGESTIONS.ranges.map((r) => (
                                                    <button
                                                        key={r}
                                                        onClick={() => setLimit(r)}
                                                        className="text-[9px] px-2 py-0.5 rounded-full bg-primary/5 border border-primary/10 text-primary/70 hover:bg-primary/20 hover:text-primary transition-all"
                                                    >
                                                        {r}
                                                    </button>
                                                ))}
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>

                            <div className="space-y-2">
                                <label className="text-[10px] uppercase tracking-[0.15em] text-label font-bold ml-1">Target Attributes</label>
                                <textarea
                                    value={attributes}
                                    onChange={(e) => setAttributes(e.target.value)}
                                    onFocus={() => setActiveField('attributes')}
                                    rows={2}
                                    className="glass-input w-full resize-none min-h-[80px]"
                                    placeholder="Enter Target Attributes (e.g. specific capacitance, pore volume)"
                                />
                                <AnimatePresence>
                                    {activeField === 'attributes' && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            exit={{ opacity: 0, height: 0 }}
                                            className="overflow-hidden"
                                        >
                                            <div className="space-y-3 mt-3 ml-1">
                                                {Object.entries(SUGGESTIONS.attributes).map(([domain, text]) => (
                                                    <div key={domain} className="space-y-1">
                                                        <div className="flex items-center gap-1.5">
                                                            <div className="h-[1px] w-2 bg-primary/30" />
                                                            <span className="text-[9px] font-black uppercase tracking-widest text-primary/60">{domain}</span>
                                                        </div>
                                                        <button
                                                            onClick={() => setAttributes(text)}
                                                            className="text-[9px] text-label/70 hover:text-primary transition-colors text-left leading-relaxed block pl-3 border-l border-primary/10"
                                                        >
                                                            {text.substring(0, 100)}...
                                                        </button>
                                                    </div>
                                                ))}
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                                <p className="text-[10px] text-label italic mt-4 ml-1 flex items-center gap-1">
                                    <Database className="w-3 h-3 text-primary/60" /> Data points to extract.
                                </p>
                            </div>

                            <div className="space-y-4 pt-4">
                                <button
                                    onClick={handleStart}
                                    disabled={status.is_running || !query.trim() || !taskName.trim()}
                                    className="premium-button w-full group"
                                >
                                    <AnimatePresence mode='wait'>
                                        {status.is_running ? (
                                            <motion.div
                                                key="loading"
                                                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                                                className="flex items-center gap-2"
                                            >
                                                <Loader2 className="w-4 h-4 animate-spin" />
                                                <span>Mining in Progress</span>
                                            </motion.div>
                                        ) : (
                                            <motion.div
                                                key="idle"
                                                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                                                className="flex items-center gap-2"
                                            >
                                                <Play className="w-4 h-4" />
                                                <span>Execute Pipeline</span>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </button>

                                {status.is_running && (
                                    <button
                                        onClick={handleStop}
                                        className="w-full py-4 rounded-2xl bg-rose-900/20 border border-rose-900/30 text-rose-500 font-bold text-sm flex items-center justify-center gap-2 hover:bg-rose-900/30 transition-colors"
                                    >
                                        <Square className="w-4 h-4 fill-current" />
                                        <span>Stop Mining Process</span>
                                    </button>
                                )}

                                {jobId && !status.is_running && (
                                    <button
                                        onClick={handleNewSession}
                                        className="w-full py-3 text-[10px] font-black uppercase tracking-widest text-label hover:text-primary transition-colors"
                                    >
                                        Start New Research Session
                                    </button>
                                )}
                            </div>
                        </div>
                    </motion.section>

                    {/* Progress Monitor */}
                    {status.is_running && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.98 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="glass-card rounded-2xl p-6 border-accent/20 bg-accent/5"
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <Activity className="w-4 h-4 text-accent animate-pulse" />
                                    <span className="text-[11px] font-black uppercase tracking-widest text-label">
                                        Current Phase: <span className="text-accent">{status.current_step}</span>
                                    </span>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {/* Terminal View */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="glass-card rounded-[2rem] h-[500px] flex flex-col p-8 overflow-hidden"
                    >
                        <header className="flex items-center justify-between mb-6">
                            <div className="flex items-center gap-3">
                                <div className="flex gap-1.5">
                                    <div className="w-3 h-3 rounded-full bg-rose-500/20 border border-rose-500/40" />
                                    <div className="w-3 h-3 rounded-full bg-amber-500/20 border border-amber-500/40" />
                                    <div className="w-3 h-3 rounded-full bg-emerald-500/20 border border-emerald-500/40" />
                                </div>
                                <span className="text-[11px] font-mono text-label uppercase tracking-widest pl-2">Console_Output</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={scrollToTop}
                                    title="Jump to Top"
                                    className="p-1.5 hover:bg-border/20 text-label hover:text-primary rounded-lg transition-all"
                                >
                                    <ArrowUp className="w-4 h-4" />
                                </button>
                                <button
                                    onClick={scrollToBottom}
                                    title="Jump to Bottom"
                                    className="p-1.5 hover:bg-border/20 text-label hover:text-primary rounded-lg transition-all"
                                >
                                    <ArrowDown className="w-4 h-4" />
                                </button>
                                <div className="h-4 w-[1px] bg-border mx-1" />
                                <button
                                    onClick={handleCopyAll}
                                    className={`text-[10px] font-bold transition-all flex items-center gap-1 px-3 py-1 rounded-full border ${
                                        isCopied 
                                        ? 'bg-emerald-500/20 text-emerald-500 border-emerald-500/50' 
                                        : 'bg-border/20 text-label hover:text-primary border-border hover:border-primary/50'
                                    }`}
                                >
                                    {isCopied ? 'Copied!' : 'Copy All'}
                                </button>
                                <Cpu className="w-4 h-4 text-label" />
                            </div>
                        </header>

                        <div
                            ref={logContainerRef}
                            className="bg-slate-950 border border-border rounded-2xl p-6 font-mono text-[11px] text-slate-300 overflow-y-auto overflow-x-auto flex-1 space-y-2 custom-scrollbar shadow-2xl select-text"
                        >
                            {status.logs.length === 0 ? (
                                <div className="flex flex-col items-center justify-center h-full space-y-4 text-slate-500">
                                    <Terminal className="w-12 h-12 opacity-20" />
                                    <p className="italic font-medium">Awaiting pipeline trigger... logs will stream here.</p>
                                </div>
                            ) : (
                                status.logs.map((log, i) => (
                                    <div key={i} className={`flex gap-4 group min-w-max ${(log as string).includes('Starting') ? 'text-primary font-black' : ''}`}>
                                        <span className="text-slate-600 w-6 shrink-0">{i + 1}</span>
                                        <span className="leading-relaxed whitespace-pre">{log}</span>
                                    </div>
                                ))
                            )}
                            <div ref={logEndRef} />
                        </div>
                    </motion.div>

                    {/* Quick Access Downloads */}
                    <div className="flex flex-col md:flex-row gap-4">
                        <button
                            onClick={() => jobId && window.open(`${API_BASE_URL}/download-pdfs/${jobId}`)}
                            disabled={!jobId || results.length === 0}
                            className="flex-1 group flex items-center justify-center gap-2.5 bg-primary hover:bg-primary/90 text-white py-4 rounded-2xl transition-all shadow-xl shadow-primary/20 font-bold text-sm active:scale-95 disabled:opacity-30"
                        >
                            <Download className="w-4 h-4" />
                            Download Project Bundle (ZIP)
                        </button>
                        <button
                            onClick={() => jobId && window.open(`${API_BASE_URL}/download/${jobId}`)}
                            disabled={!jobId || results.length === 0}
                            className="flex-1 group flex items-center justify-center gap-2.5 bg-slate-900 hover:bg-black text-white py-4 rounded-2xl transition-all shadow-xl shadow-slate-900/10 font-bold text-sm active:scale-95 disabled:opacity-30"
                        >
                            <Download className="w-4 h-4" />
                            Export Dataset (CSV)
                        </button>
                    </div>

                    {/* Excel-Style Dataset Viewer */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="glass-card rounded-[2rem] p-8 overflow-hidden h-[600px] flex flex-col"
                    >
                        <header className="flex justify-between items-center mb-6">
                            <div className="space-y-1">
                                <h2 className="text-2xl font-black tracking-tight text-foreground uppercase flex items-center gap-2">
                                    <Database className="w-6 h-6 text-primary" /> Extraction Results
                                </h2>
                                <p className="text-xs text-label font-medium uppercase tracking-wide">
                                    {results.length} records retrieved from source documents.
                                </p>
                            </div>
                        </header>

                        <div className="flex-1 overflow-auto rounded-2xl border border-slate-200 bg-white shadow-inner custom-scrollbar">
                            <table className="min-w-full divide-y divide-slate-200 border-collapse table-auto">
                                <thead className="bg-slate-50 sticky top-0 z-10">
                                    <tr className="divide-x divide-slate-200">
                                        <th className="px-4 py-3 text-left text-[10px] font-black uppercase tracking-widest text-slate-500 bg-slate-50 border-b border-slate-200 min-w-[200px]">
                                            Source Document
                                        </th>
                                        {results.length > 0 && Object.keys(results[0])
                                            .filter(k => !['File', 'Title', 'source_file', 'filename', 'source', 'formula_used'].includes(k) && !k.endsWith('_flag') && !k.includes('reasons'))
                                            .map(key => (
                                                <th key={key} className="px-4 py-3 text-left text-[10px] font-black uppercase tracking-widest text-slate-500 border-b border-slate-200 min-w-[120px]">
                                                    {key.replace(/_/g, ' ')}
                                                </th>
                                            ))
                                        }
                                        <th className="px-4 py-3 text-right text-[10px] font-black uppercase tracking-widest text-slate-500 border-b border-slate-200">
                                            View
                                        </th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-slate-100 font-mono text-[11px]">
                                    {results.length === 0 ? (
                                        <tr>
                                            <td colSpan={50} className="px-6 py-24 text-center">
                                                <div className="flex flex-col items-center gap-3 text-slate-800">
                                                    <Database className="w-12 h-12" />
                                                    <span className="font-bold underline italic">Dataset Empty</span>
                                                </div>
                                            </td>
                                        </tr>
                                    ) : (
                                        results.map((row, i) => (
                                            <tr key={i} className="divide-x divide-slate-50 hover:bg-slate-50/50 transition-colors">
                                                <td className="px-4 py-3 font-bold text-slate-900 border-r border-slate-100 truncate max-w-[300px]">
                                                    {row.File || row.Title || "Unknown"}
                                                </td>
                                                {Object.keys(results[0])
                                                    .filter(k => !['File', 'Title', 'source_file', 'filename', 'source', 'formula_used'].includes(k) && !k.endsWith('_flag') && !k.includes('reasons'))
                                                    .map(key => (
                                                        <td key={key} className="px-4 py-3 text-slate-600 border-r border-slate-50">
                                                            {row[key] !== null && row[key] !== undefined && row[key] !== '' ? row[key] : '—'}
                                                        </td>
                                                    ))
                                                }
                                                <td className="px-4 py-3 text-right">
                                                    {row.File && (
                                                        <a
                                                            href={`${API_BASE_URL}/pdf/${jobId}/${row.File}`}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            className="p-1.5 hover:bg-primary/10 text-slate-400 hover:text-primary rounded-lg transition-all inline-block"
                                                        >
                                                            <ExternalLink className="w-3.5 h-3.5" />
                                                        </a>
                                                    )}
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </motion.div>
                </div>

                {/* Footer Insight */}
                <footer className="pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center text-[11px] text-label font-black uppercase tracking-[0.2em] gap-4">
                    <div className="flex items-center gap-6">
                        <span>Cluster: 172.168.14.28_8001</span>
                        <span>Protocol: Academic_API_V5</span>
                    </div>
                    <div className="flex items-center gap-1">
                        Designed for <Sparkles className="w-3 h-3 text-primary mx-1" /> Scientific Discovery
                    </div>
                </footer>
            </main>
        </div>
    );
};

export default Dashboard;