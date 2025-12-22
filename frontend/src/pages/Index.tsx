import { useState, useRef, useEffect } from "react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, Send, Info, ChevronDown, Scale, Cog, Database, AlertTriangle, Sun, Moon } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { API_URL } from "@/lib/utils";

interface Message {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: Date;
  showContext?: boolean;
  context?: string[];
}


// Abstracted API call - can be replaced with deployed URL
interface ApiResponse {
  answer: string;
  show_context?: boolean;
  context?: string[];
}

const askQuestion = async (question: string): Promise<ApiResponse> => {
  const response = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question: question.trim() }),
  });

  if (!response.ok) {
    throw new Error(`Server responded with status ${response.status}`);
  }

  return response.json();
};



const Index = () => {
  const { theme, setTheme } = useTheme();
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async () => {
    if (!question.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: question.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await askQuestion(question);
      const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: "assistant",
      content: response.answer,
      timestamp: new Date(),
      showContext: response.show_context,
      context: response.context,
    };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to get a response. Please try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Compact Header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="container max-w-3xl py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Scale className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h1 className="font-heading text-xl font-bold text-foreground tracking-tight">
                  Municipal Law Assistant
                </h1>
                <p className="text-muted-foreground text-xs font-body">
                  RAG-based Question Answering System
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-1">
              {/* Theme Toggle */}
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={toggleTheme}
                className="text-muted-foreground hover:text-foreground"
              >
                {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </Button>
              
              {/* How It Works Modal */}
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
                    <Cog className="h-4 w-4" />
                    <span className="hidden sm:inline">How It Works</span>
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle className="font-heading flex items-center gap-2">
                      <Cog className="h-5 w-5 text-primary" />
                      How It Works
                    </DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4 text-sm text-muted-foreground font-body">
                    <div className="space-y-3">
                      <div className="flex items-start gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-semibold flex items-center justify-center">1</span>
                        <div>
                          <p className="font-medium text-foreground">Query Preprocessing</p>
                          <p className="text-xs mt-0.5">Text normalization, tokenization, and cleaning</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-semibold flex items-center justify-center">2</span>
                        <div>
                          <p className="font-medium text-foreground">TF-IDF Based Retrieval</p>
                          <p className="text-xs mt-0.5">Retrieve relevant documents from the corpus</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-semibold flex items-center justify-center">3</span>
                        <div>
                          <p className="font-medium text-foreground">Custom Neural Text Generation</p>
                          <p className="text-xs mt-0.5">Generate answers using trained neural model</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-semibold flex items-center justify-center">4</span>
                        <div>
                          <p className="font-medium text-foreground">Fallback Summarization</p>
                          <p className="text-xs mt-0.5">Extractive summary when generation is uncertain</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>

              {/* Dataset & Training Modal */}
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
                    <Database className="h-4 w-4" />
                    <span className="hidden sm:inline">Dataset</span>
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle className="font-heading flex items-center gap-2">
                      <Database className="h-5 w-5 text-primary" />
                      Dataset & Training
                    </DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4 text-sm text-muted-foreground font-body">
                    <ul className="space-y-3">
                      <li className="flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                        <span>Municipal law documents collected manually from official sources</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                        <span>300–500 custom question–answer pairs curated for training</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                        <span>Vocabulary built from scratch using domain-specific terms</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                        <span>No pretrained models or external AI APIs used</span>
                      </li>
                    </ul>
                  </div>
                </DialogContent>
              </Dialog>

              {/* Limitations Modal */}
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="hidden sm:inline">Limitations</span>
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle className="font-heading flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5 text-amber-500" />
                      Limitations
                    </DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4 text-sm text-muted-foreground font-body">
                    <ul className="space-y-3">
                      <li className="flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-2 flex-shrink-0" />
                        <span>Domain-specific knowledge limited to municipal laws only</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-2 flex-shrink-0" />
                        <span>Limited training data (300–500 QA pairs)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-2 flex-shrink-0" />
                        <span>Designed for academic demonstration purposes</span>
                      </li>
                    </ul>
                    <div className="pt-3 border-t border-border">
                      <p className="text-xs text-muted-foreground/80">
                        This system is intended for educational evaluation and should not be used for legal advice.
                      </p>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>

              {/* About Modal */}
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
                    <Info className="h-4 w-4" />
                    <span className="hidden sm:inline">About</span>
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle className="font-heading">About This System</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-3 text-sm text-muted-foreground font-body">
                    <p>
                      This system uses classical NLP, retrieval, and a custom-trained 
                      neural model. No pretrained models or external AI APIs are used.
                    </p>
                    <div className="pt-2 border-t border-border">
                      <p className="text-xs">
                        NLP Project · Municipal Law Assistant
                      </p>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 container max-w-3xl py-6 flex flex-col">
        <div className="flex-1 space-y-4 pb-4 overflow-y-auto">
          {/* Welcome State */}
          {messages.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center h-full min-h-[300px] text-center px-4">
              <div className="p-4 rounded-2xl bg-accent/50 mb-4">
                <Scale className="h-10 w-10 text-primary" />
              </div>
              <h2 className="font-heading text-2xl font-semibold text-foreground mb-2">
                Welcome to Municipal Law Assistant
              </h2>
              <p className="text-muted-foreground font-body max-w-md">
                Ask questions about municipal laws, regulations, and procedures. 
                The system will retrieve relevant legal context and generate answers.
              </p>
              <div className="mt-6 flex flex-wrap gap-2 justify-center">
                {[
                  "What are the penalties for illegal waste disposal?",
                  "Building permit requirements",
                  "Noise ordinance regulations",
                ].map((example) => (
                  <button
                    key={example}
                    onClick={() => setQuestion(example)}
                    className="px-3 py-1.5 text-sm rounded-full bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors font-body"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] ${
                  message.type === "user"
                    ? "bg-primary text-primary-foreground rounded-2xl rounded-br-md px-4 py-3"
                    : "space-y-2"
                }`}
              >
                {message.type === "assistant" && (
                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Generated Answer (Custom NLP Model)
                  </span>
                )}
                <div
                  className={
                    message.type === "assistant"
                      ? "bg-card border border-border rounded-2xl rounded-bl-md px-4 py-3 shadow-card"
                      : ""
                  }
                >
                  <p className="font-body text-sm leading-relaxed whitespace-pre-wrap">
                    {message.content}
                  </p>
                </div>
                
                {/* Retrieved Context - Collapsible (for future use) */}
                {message.type === "assistant" && message.showContext && (
                  <Collapsible>
                    <CollapsibleTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors mt-2 group">
                      <ChevronDown className="h-3 w-3 transition-transform group-data-[state=open]:rotate-180" />
                      Retrieved Legal Context
                    </CollapsibleTrigger>

                    <CollapsibleContent className="mt-2">
                      <div className="bg-muted/50 rounded-lg px-3 py-2 text-xs text-muted-foreground font-body space-y-1">
                        {message.context?.map((c, i) => (
                          <p key={i}>• {c}</p>
                        ))}
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                )}
              </div>
            </div>
          ))}

          {/* Loading State */}
          {isLoading && (
            <div className="flex justify-start">
              <div className="space-y-2 max-w-[85%]">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Analyzing municipal laws...
                </span>
                <div className="bg-card border border-border rounded-2xl rounded-bl-md px-4 py-3 shadow-card">
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                      <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                      <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                    <span className="text-sm text-muted-foreground font-body">
                      Retrieving relevant context...
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="flex justify-start">
              <div className="max-w-[85%] bg-destructive/10 border border-destructive/20 rounded-2xl rounded-bl-md px-4 py-3">
                <p className="text-destructive font-body text-sm">{error}</p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area - Fixed at Bottom */}
        <div className="sticky bottom-0 pt-4 bg-gradient-to-t from-background via-background to-transparent">
          <div className="bg-card border border-border rounded-2xl shadow-card p-3">
            <Textarea
              ref={textareaRef}
              placeholder="Ask a question about municipal laws (e.g., waste management penalties)"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              className="min-h-[60px] max-h-[150px] text-sm font-body resize-none border-0 focus-visible:ring-0 bg-transparent p-0 placeholder:text-muted-foreground/60"
              disabled={isLoading}
            />
            <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/50">
              <span className="text-xs text-muted-foreground">
                Press <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono">Ctrl</kbd> + <kbd className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-mono">Enter</kbd> to submit
              </span>
              <Button
                onClick={handleSubmit}
                disabled={isLoading || !question.trim()}
                size="sm"
                className="gap-2 rounded-xl"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <>
                    <Send className="h-4 w-4" />
                    Send
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
