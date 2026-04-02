"use client"

import React from "react"

import { useState, useEffect } from "react"
import { PageLayout, StaggerContainer, StaggerItem } from "@/components/page-layout"
import {
  uploadFile,
  startTraining,
  checkHealth,
  testConnection,
  listUploadedFiles,
  deleteUploadedFile,
  submitFeedback,
  getPublicAnswers,
  getFeedbackMessages,
  createFeedbackFollowUp,
  type HealthResponse,
  type TrainingResult,
  type PublicAnswerItem,
  type ConversationMessageItem,
} from "@/lib/api"
import { SectionHeader } from "@/components/section-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Upload,
  Play,
  Wind,
  TrendingUp,
  Clock,
  FileText,
  AlertCircle,
  CheckCircle2,
  BarChart3,
  Shield,
  Server,
  Layers,
  Trash2,
  RefreshCw,
  MessageSquare,
  Send,
} from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
} from "recharts"

// Mock data for wind speed demonstration (removed power data)
const mockForecastData = [
  { hour: "00:00", actual: 8.2, predicted: 8.0 },
  { hour: "02:00", actual: 9.1, predicted: 8.8 },
  { hour: "04:00", actual: 7.8, predicted: 8.1 },
  { hour: "06:00", actual: 10.5, predicted: 10.2 },
  { hour: "08:00", actual: 12.3, predicted: 11.9 },
  { hour: "10:00", actual: 11.8, predicted: 12.1 },
  { hour: "12:00", actual: 13.5, predicted: 13.2 },
  { hour: "14:00", actual: 14.2, predicted: 14.5 },
  { hour: "16:00", actual: 12.8, predicted: 13.0 },
  { hour: "18:00", actual: 10.5, predicted: 10.8 },
  { hour: "20:00", actual: 9.2, predicted: 9.0 },
  { hour: "22:00", actual: 8.5, predicted: 8.7 },
]

export default function DashboardPage() {
  // Form state
  const [files, setFiles] = useState<File[]>([])
  const [predictionLength, setPredictionLength] = useState("6")
  const [trainingModel, setTrainingModel] = useState("GPT4TS")
  const [federalAlgorithm, setFederalAlgorithm] = useState("FedAvg")
  const [numClients, setNumClients] = useState("5")
  const [numRounds, setNumRounds] = useState("5")
  const [localEpochs, setLocalEpochs] = useState("1")
  const [llmLayers, setLlmLayers] = useState("4")
  const [dropoutRate, setDropoutRate] = useState("0.2")
  const [mode, setMode] = useState("federated")
  const [horizon, setHorizon] = useState("6") // Declared horizon state

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingStep, setProcessingStep] = useState("")
  const [showResults, setShowResults] = useState(false)

  // API state
  const [apiConnected, setApiConnected] = useState<boolean | null>(null)
  const [apiHealth, setApiHealth] = useState<HealthResponse | null>(null)
  const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  // File management state
  const [uploadedFiles, setUploadedFiles] = useState<Array<{filename: string, size: number, modified: string}>>([])
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [deletingFile, setDeletingFile] = useState<string | null>(null)

  // Feedback state
  const [feedbackMessage, setFeedbackMessage] = useState("")
  const [feedbackName, setFeedbackName] = useState("")
  const [sendingFeedback, setSendingFeedback] = useState(false)
  const [feedbackSent, setFeedbackSent] = useState(false)

  // Public answers state
  const [publicAnswers, setPublicAnswers] = useState<PublicAnswerItem[]>([])
  const [loadingPublicAnswers, setLoadingPublicAnswers] = useState(false)
  const [currentAnswersPage, setCurrentAnswersPage] = useState(1)
  const [expandedConversationId, setExpandedConversationId] = useState<string | null>(null)
  const [loadingConversationId, setLoadingConversationId] = useState<string | null>(null)
  const [conversationMessages, setConversationMessages] = useState<Record<string, ConversationMessageItem[]>>({})
  const [replyConversationId, setReplyConversationId] = useState<string | null>(null)
  const [replyToMessageId, setReplyToMessageId] = useState<string | null>(null)
  const [followUpMessage, setFollowUpMessage] = useState("")
  const [followUpName, setFollowUpName] = useState("")
  const [sendingFollowUp, setSendingFollowUp] = useState(false)
  const [expandedReplyChildren, setExpandedReplyChildren] = useState<Record<string, boolean>>({})
  const ANSWERS_PER_PAGE = 5

  // Load uploaded files list
  const loadUploadedFiles = async () => {
    console.log("[v0] loadUploadedFiles called, apiConnected:", apiConnected)
    if (!apiConnected) {
      console.log("[v0] API not connected, skipping file load")
      return
    }
    setLoadingFiles(true)
    try {
      console.log("[v0] Fetching uploaded files from API...")
      const result = await listUploadedFiles()
      console.log("[v0] Files loaded:", result)
      setUploadedFiles(result.files)
    } catch (err) {
      console.error("[v0] Failed to load files:", err)
    } finally {
      setLoadingFiles(false)
    }
  }

  // Submit feedback
  const handleSubmitFeedback = async () => {
    if (!feedbackMessage.trim()) return
    
    setSendingFeedback(true)
    try {
      await submitFeedback(
        feedbackMessage.trim(),
        feedbackName.trim() || undefined,
        "dashboard",
      )
      setFeedbackMessage("")
      setFeedbackName("")
      setFeedbackSent(true)
      setTimeout(() => setFeedbackSent(false), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send feedback")
    } finally {
      setSendingFeedback(false)
    }
  }

  const loadPublicAnswers = async () => {
    if (!apiConnected) return

    setLoadingPublicAnswers(true)
    try {
      const result = await getPublicAnswers()
      setPublicAnswers(result.entries)
      setCurrentAnswersPage(1)
    } catch (err) {
      console.error("[v0] Failed to load public answers:", err)
    } finally {
      setLoadingPublicAnswers(false)
    }
  }

  const loadConversationMessages = async (conversationId: string) => {
    setLoadingConversationId(conversationId)
    try {
      const result = await getFeedbackMessages(conversationId)
      setConversationMessages((prev) => ({
        ...prev,
        [conversationId]: result.entries,
      }))
      return result.entries
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load conversation")
      return []
    } finally {
      setLoadingConversationId(null)
    }
  }

  const toggleConversationThread = async (conversationId: string) => {
    if (expandedConversationId === conversationId) {
      setExpandedConversationId(null)

      if (replyConversationId === conversationId) {
        setReplyConversationId(null)
        setReplyToMessageId(null)
        setFollowUpMessage("")
        setFollowUpName("")
      }
      return
    }

    setExpandedConversationId(conversationId)

    if (!conversationMessages[conversationId]) {
      await loadConversationMessages(conversationId)
    }
  }

  const getVisibleMessages = (conversationId: string) => {
    const messages = conversationMessages[conversationId] ?? []
    return messages.filter(
      (message) => message.sender_type === "user" || message.is_public,
    )
  }

  const getThreadRootQuestionMessage = (conversationId: string) => {
    const messages = conversationMessages[conversationId] ?? []
    return messages.find((message) => message.sender_type === "user") ?? null
  }

  const getMainOfficialAnswerMessage = (conversationId: string) => {
    const rootQuestion = getThreadRootQuestionMessage(conversationId)
    if (!rootQuestion) return null

    const messages = conversationMessages[conversationId] ?? []
    const matches = messages
      .filter(
        (message) =>
          message.sender_type === "developer" &&
          message.is_public &&
          message.reply_to_message_id === rootQuestion.id,
      )
      .sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      )

    return matches.length > 0 ? matches[matches.length - 1] : null
  }

  const findMessageById = (conversationId: string, messageId: string | null) => {
    if (!messageId) return null
    const messages = conversationMessages[conversationId] ?? []
    return messages.find((message) => message.id === messageId) ?? null
  }

  const getMessageElementId = (messageId: string) => `thread-message-${messageId}`

  const scrollToMessage = (messageId: string | null) => {
    if (!messageId) return

    const element = document.getElementById(getMessageElementId(messageId))
    if (!element) return

    element.scrollIntoView({ behavior: "smooth", block: "center" })

    element.classList.add(
      "rounded-md",
      "bg-yellow-100/60",
      "ring-1",
      "ring-yellow-300/60",
      "transition-all",
      "duration-300"
    )

    window.setTimeout(() => {
      element.classList.remove(
        "rounded-md",
        "bg-yellow-100/60",
        "ring-1",
        "ring-yellow-300/60",
        "transition-all",
        "duration-300"
      )
    }, 900)
  }

  const getReplyPreviewText = (conversationId: string, replyToMessageId: string | null) => {
    const targetMessage = findMessageById(conversationId, replyToMessageId)
    if (!targetMessage) return "Earlier message"

    const text = targetMessage.message_text.trim()
    if (text.length <= 60) return text
    return `${text.slice(0, 60)}...`
  }

  const openReplyFormForQuestion = async (conversationId: string) => {
    let messages = conversationMessages[conversationId]

    if (!messages) {
      setExpandedConversationId(conversationId)
      messages = await loadConversationMessages(conversationId)
    } else if (expandedConversationId !== conversationId) {
      setExpandedConversationId(conversationId)
    }

    const rootQuestion =
      messages?.find((message) => message.sender_type === "user") ?? null

    if (!rootQuestion) {
      setError("Could not find the main question to reply to.")
      return
    }

    setReplyConversationId(conversationId)
    setReplyToMessageId(rootQuestion.id)
  }

  const openReplyFormForMessage = async (conversationId: string, messageId: string) => {
    if (expandedConversationId !== conversationId) {
      setExpandedConversationId(conversationId)
    }

    if (!conversationMessages[conversationId]) {
      await loadConversationMessages(conversationId)
    }

    setReplyConversationId(conversationId)
    setReplyToMessageId(messageId)
  }

  const closeReplyForm = () => {
    setReplyConversationId(null)
    setReplyToMessageId(null)
    setFollowUpMessage("")
    setFollowUpName("")
  }

  const handleSubmitFollowUp = async () => {
    if (!replyConversationId || !replyToMessageId || !followUpMessage.trim()) return

    setSendingFollowUp(true)
    try {
      await createFeedbackFollowUp(
        replyConversationId,
        followUpMessage.trim(),
        followUpName.trim() || undefined,
        "dashboard",
        replyToMessageId,
      )

      const conversationId = replyConversationId
      closeReplyForm()
      setExpandedConversationId(conversationId)
      await loadConversationMessages(conversationId)
      await loadPublicAnswers()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send follow-up")
    } finally {
      setSendingFollowUp(false)
    }
  }

  const getTopLevelThreadMessages = (conversationId: string) => {
    const visibleMessages = getVisibleMessages(conversationId)
    const rootQuestion = getThreadRootQuestionMessage(conversationId)
    const mainOfficialAnswer = getMainOfficialAnswerMessage(conversationId)

    if (!rootQuestion) return []

    return visibleMessages
      .filter((message) => {
        if (message.id === rootQuestion.id) return false
        if (mainOfficialAnswer && message.id === mainOfficialAnswer.id) return false
        if (!message.reply_to_message_id) return false

        return (
          message.reply_to_message_id === rootQuestion.id ||
          (mainOfficialAnswer && message.reply_to_message_id === mainOfficialAnswer.id)
        )
      })
      .sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      )
  }

  const getChildThreadMessages = (conversationId: string, parentMessageId: string) => {
    const visibleMessages = getVisibleMessages(conversationId)

    return visibleMessages
      .filter((message) => message.reply_to_message_id === parentMessageId)
      .sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      )
  }

  const getOrphanPublicReplies = (conversationId: string) => {
    const visibleMessages = getVisibleMessages(conversationId)
    const rootQuestion = getThreadRootQuestionMessage(conversationId)
    const mainOfficialAnswer = getMainOfficialAnswerMessage(conversationId)
    const validParentIds = new Set(visibleMessages.map((message) => message.id))

    return visibleMessages.filter((message) => {
      if (!(message.sender_type === "developer" && message.is_public)) {
        return false
      }

      if (!message.reply_to_message_id) {
        return false
      }

      if (rootQuestion && message.reply_to_message_id === rootQuestion.id) {
        return false
      }

      if (mainOfficialAnswer && message.reply_to_message_id === mainOfficialAnswer.id) {
        return false
      }

      return !validParentIds.has(message.reply_to_message_id)
    })
  }

  const toggleReplyChildren = (messageId: string) => {
    setExpandedReplyChildren((prev) => ({
      ...prev,
      [messageId]: !prev[messageId],
    }))
  } 


  const renderReplyForm = (conversationId: string, messageId: string) => {
    const isOpen =
      replyConversationId === conversationId && replyToMessageId === messageId

    if (!isOpen) return null

    return (
      <div className="mt-3 rounded-lg border border-border bg-background p-4 space-y-4">
        <div>
          <label className="text-sm font-medium text-foreground block mb-2">
            Your Name (optional)
          </label>
          <input
            type="text"
            value={followUpName}
            onChange={(e) => setFollowUpName(e.target.value)}
            placeholder="Enter your name or leave blank for anonymous"
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
          />
        </div>

        <div>
          <label className="text-sm font-medium text-foreground block mb-2">
            Reply
          </label>
          <textarea
            value={followUpMessage}
            onChange={(e) => setFollowUpMessage(e.target.value)}
            placeholder="Write your reply here..."
            className="w-full rounded-md border border-input bg-background px-3 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 resize-vertical min-h-24"
          />
        </div>

        <div className="flex justify-end gap-2">
          <Button
            variant="outline"
            onClick={closeReplyForm}
            disabled={sendingFollowUp}
          >
            Cancel
          </Button>

          <Button
            onClick={handleSubmitFollowUp}
            disabled={sendingFollowUp || !followUpMessage.trim()}
            className="gap-2"
          >
            {sendingFollowUp ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-background border-t-transparent" />
                Sending...
              </>
            ) : (
              <>
                <Send className="h-4 w-4" />
                Send Reply
              </>
            )}
          </Button>
        </div>
      </div>
    )
  }

const renderConversationNode = (
  conversationId: string,
  message: ConversationMessageItem,
  depth = 0,
): React.ReactNode => {
  const childMessages = getChildThreadMessages(conversationId, message.id)
  const hasChildren = childMessages.length > 0
  const showChildren = expandedReplyChildren[message.id] ?? false

  return (
    <div
      key={message.id}
      className={depth > 0 ? "ml-6 border-l border-border pl-4 space-y-3" : "space-y-3"}
    >
      <div
        id={getMessageElementId(message.id)}
        className={`rounded-md border p-3 ${
          message.sender_type === "developer"
            ? "border-primary/20 bg-primary/5"
            : "border-border bg-muted/20"
        }`}
      >
        <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span>
            {message.sender_type === "developer"
              ? "Official Reply"
              : message.sender_name || "User"}{" "}
            • {new Date(message.created_at).toLocaleString()}
          </span>

          {message.reply_to_message_id && (
            <button
              type="button"
              onClick={() => scrollToMessage(message.reply_to_message_id)}
              className="italic underline underline-offset-2 hover:text-foreground"
            >
              Re: {getReplyPreviewText(conversationId, message.reply_to_message_id)}
            </button>
          )}
        </div>

        <p className="text-sm text-foreground whitespace-pre-wrap">
          {message.message_text}
        </p>

        <div className="mt-2 flex flex-wrap items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => openReplyFormForMessage(conversationId, message.id)}
            className="h-auto p-0 text-xs font-bold text-muted-foreground hover:bg-transparent hover:text-foreground"
          >
            Reply
          </Button>

          {hasChildren && (
            <button
              type="button"
              onClick={() => toggleReplyChildren(message.id)}
              className="text-xs font-medium text-muted-foreground underline underline-offset-2 hover:text-foreground"
            >
              {showChildren
                ? `Hide replies (${childMessages.length})`
                : `Show replies (${childMessages.length})`}
            </button>
          )}
        </div>

        {renderReplyForm(conversationId, message.id)}
      </div>

      {hasChildren && showChildren && (
        <div className="space-y-3">
          {childMessages.map((childMessage) =>
            renderConversationNode(conversationId, childMessage, depth + 1),
          )}
        </div>
      )}
    </div>
  )
}


  const totalAnswerPages = Math.max(1, Math.ceil(publicAnswers.length / ANSWERS_PER_PAGE))
  const startAnswerIndex = (currentAnswersPage - 1) * ANSWERS_PER_PAGE
  const endAnswerIndex = startAnswerIndex + ANSWERS_PER_PAGE
  const visiblePublicAnswers = publicAnswers.slice(startAnswerIndex, endAnswerIndex)

  const answerPageNumbers = Array.from({ length: totalAnswerPages }, (_, i) => i + 1)

  // Delete a file
  const handleDeleteFile = async (filename: string) => {
    console.log("[v0] Delete requested for file:", filename)
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
      console.log("[v0] Delete cancelled by user")
      return
    }
    
    setDeletingFile(filename)
    try {
      console.log("[v0] Calling deleteUploadedFile API...")
      const result = await deleteUploadedFile(filename)
      console.log("[v0] Delete result:", result)
      // Reload file list
      console.log("[v0] Reloading file list after delete...")
      await loadUploadedFiles()
    } catch (err) {
      console.error("[v0] Delete failed:", err)
      setError(err instanceof Error ? err.message : "Failed to delete file")
    } finally {
      setDeletingFile(null)
    }
  }

  // Check API connection on mount
  useEffect(() => {  
    const checkApiConnection = async () => {
      try {
        const connected = await testConnection()
        setApiConnected(connected)
        if (connected) {
          const health = await checkHealth()
          setApiHealth(health)
        }
      } catch {
        setApiConnected(false)
      }
    }
    checkApiConnection()
  }, [])

  useEffect(() => {
    if (apiConnected) {
      loadUploadedFiles()
      loadPublicAnswers()
    }
  }, [apiConnected])

  useEffect(() => {
    if (currentAnswersPage > totalAnswerPages) {
      setCurrentAnswersPage(totalAnswerPages)
    }
  }, [currentAnswersPage, totalAnswerPages])

  // File upload handler
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files)
      setFiles(prev => [...prev, ...newFiles])
      setError(null)
    }
  }

  // Refresh Answers handler
  const handleRefreshAnswers = async () => {
    await loadPublicAnswers()

    if (expandedConversationId) {
      await loadConversationMessages(expandedConversationId)
    }
  }

  // Remove a file from the selection
  const handleRemoveFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  // Training/upload handler - connects to FastAPI backend
  const handleStartTraining = async () => {
    setIsProcessing(true)
    setError(null)
    setTrainingResult(null)

    try {
      if (mode === "centralized") {
        // Centralized mode: Upload files -> then train on last one
        if (files.length === 0) {
          setError("Please select at least one CSV file to upload")
          return
        }

        // Step 1: Upload all files, keep track of saved filenames
        const savedFilenames: string[] = []
        for (let i = 0; i < files.length; i++) {
          const file = files[i]
          setProcessingStep(`Uploading file ${i + 1} of ${files.length}: ${file.name}...`)
          const uploadResult = await uploadFile(file)
          savedFilenames.push(uploadResult.file.filename)
        }

        // Reload file list after uploads
        await loadUploadedFiles()

        // Step 2: Train on the last uploaded file with config from the form
        const trainFilename = savedFilenames[savedFilenames.length - 1]
        setProcessingStep(
          `Training ${trainingModel} model (${predictionLength}-step horizon)...`
        )

        const config = {
          training_model: trainingModel,
          prediction_length: parseInt(predictionLength),
          dropout_rate: parseFloat(dropoutRate),
          mode: "centralized" as const,
          federated_algorithm: federalAlgorithm,
          num_clients: parseInt(numClients),
          num_rounds: parseInt(numRounds),
          local_epochs: parseInt(localEpochs),
          llm_layers: parseInt(llmLayers),
        }

        const result = await startTraining(trainFilename, config)
        setTrainingResult(result)
        setShowResults(true)

        // Reset file selection
        setFiles([])
        const fileInput = document.getElementById("file-upload") as HTMLInputElement
        if (fileInput) fileInput.value = ""
      } else {
        // Federated mode: upload CSV → server runs real flwr simulation → results returned
        if (files.length === 0) {
          setError("Please select at least one CSV file to upload")
          return
        }

        // Step 1: Upload file
        const savedFilenames: string[] = []
        for (let i = 0; i < files.length; i++) {
          const file = files[i]
          setProcessingStep(`Uploading file ${i + 1} of ${files.length}: ${file.name}...`)
          const uploadResult = await uploadFile(file)
          savedFilenames.push(uploadResult.file.filename)
        }

        await loadUploadedFiles()

        // Step 2: Run real federated simulation via flwr on the server
        const trainFilename = savedFilenames[savedFilenames.length - 1]
        setProcessingStep(
          `Running federated simulation: ${trainingModel} / ${federalAlgorithm} / ${numClients} clients...`
        )

        const config = {
          training_model: trainingModel,
          prediction_length: parseInt(predictionLength),
          dropout_rate: parseFloat(dropoutRate),
          mode: "federated" as const,
          federated_algorithm: federalAlgorithm,
          num_clients: parseInt(numClients),
          num_rounds: parseInt(numRounds),
          local_epochs: parseInt(localEpochs),
          llm_layers: parseInt(llmLayers),
        }

        const result = await startTraining(trainFilename, config)
        setTrainingResult(result)
        setShowResults(true)

        setFiles([])
        const fileInput = document.getElementById("file-upload") as HTMLInputElement
        if (fileInput) fileInput.value = ""
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsProcessing(false)
      setProcessingStep("")
    }
  }

  return (
    <PageLayout>
      {/* Header */}
      <section className="bg-gradient-to-b from-primary/5 to-background">
        <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 sm:py-16 lg:px-8">
          <div className="mx-auto max-w-3xl text-center">
            <SectionHeader
              badge="Interactive Demo"
              title="Wind Speed Forecasting Dashboard"
              description="Upload your wind data, configure training parameters, and view forecasting results. Choose between centralized or privacy-preserving federated learning."
              centered
            />
          </div>
        </div>
      </section>

      {/* API Connection Status */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div
          className={`rounded-lg border p-4 flex items-start gap-3 ${
            apiConnected === null
              ? "border-border bg-muted/30"
              : apiConnected
                ? "border-green-200 bg-green-50"
                : "border-amber-200 bg-amber-50"
          }`}
        >
          {apiConnected === null ? (
            <>
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground">Checking API Connection...</h4>
                <p className="text-sm text-muted-foreground">
                  Attempting to connect to FastAPI backend at localhost:8000
                </p>
              </div>
            </>
          ) : apiConnected ? (
            <>
              <CheckCircle2 className="h-5 w-5 text-green-600 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-green-800">API Connected</h4>
                <p className="text-sm text-green-700">
                  Backend is running (v{apiHealth?.version || "unknown"}). Model service:{" "}
                  {apiHealth?.services.model || "unknown"}
                </p>
              </div>
            </>
          ) : (
            <>
              <AlertCircle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-amber-800">Demo Mode - API Not Connected</h4>
                <p className="text-sm text-amber-700">
                  FastAPI backend not detected. Start the backend with:{" "}
                  <code className="bg-amber-100 px-1 rounded">uvicorn main:app --reload</code>
                </p>
              </div>
            </>
          )}
        </div>
      </section>

      {/* Error Display */}
      {error && (
        <section className="mx-auto max-w-7xl px-4 pt-4 sm:px-6 lg:px-8">
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-600 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-red-800">Error</h4>
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </section>
      )}

      {/* Training Success banner */}
{trainingResult && (
  <section className="mx-auto max-w-7xl px-4 pt-4 sm:px-6 lg:px-8">
    <div className="rounded-lg border border-green-200 bg-green-50 p-4 flex items-start gap-3">
      <CheckCircle2 className="h-5 w-5 text-green-600 shrink-0 mt-0.5" />
      <div>
        <h4 className="font-semibold text-green-800">Training Complete</h4>
        <p className="text-sm text-green-700">
          {trainingResult.message} | MAE: {trainingResult.metrics.mae} m/s |{" "}
          RMSE: {trainingResult.metrics.rmse} m/s |{" "}
          Time: {trainingResult.training_time_seconds}s
        </p>
      </div>
    </div>
  </section>
)}



      {/* Training Progress (shown during processing) */}
      {isProcessing && (
        <section className="mx-auto max-w-7xl px-4 pt-4 sm:px-6 lg:px-8">
          <Card className="border-primary/20 bg-primary/5">
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
                <div className="flex-1">
                  <h4 className="font-semibold text-foreground">
                    {mode === "federated" ? "Federated Training in Progress..." : "Centralized Training in Progress..."}
                  </h4>
                  <p className="text-sm text-muted-foreground">{processingStep}</p>
                </div>
              </div>
              <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-muted">
                <div className="h-full w-2/3 animate-pulse rounded-full bg-primary" />
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {mode === "federated"
                  ? "Your data stays on your device. Only model updates are shared with the server."
                  : "Your data is being processed on the server for model training."}
              </p>
            </CardContent>
          </Card>
        </section>
      )}

      {/* File Manager Section - Moved higher for better visibility */}
      {apiConnected && (
        <section className="mx-auto max-w-7xl px-4 pt-4 sm:px-6 lg:px-8">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-primary" />
                    Uploaded Files
                  </CardTitle>
                  <CardDescription>
                    Manage your uploaded wind data files (used in centralized training)
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={loadUploadedFiles}
                  disabled={loadingFiles}
                  className="gap-2 bg-transparent"
                >
                  <RefreshCw className={`h-4 w-4 ${loadingFiles ? "animate-spin" : ""}`} />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {loadingFiles ? (
                <div className="flex items-center justify-center py-8 text-muted-foreground">
                  <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent mr-3" />
                  Loading files...
                </div>
              ) : uploadedFiles.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <FileText className="h-12 w-12 text-muted-foreground mb-3" />
                  <h4 className="font-semibold text-foreground">No Files Uploaded</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Upload CSV files using the form below to get started.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {uploadedFiles.map((file) => (
                    <div
                      key={file.filename}
                      className="flex items-center justify-between rounded-lg border border-border bg-muted/20 p-3 hover:bg-muted/40 transition-colors"
                    >
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <FileText className="h-5 w-5 text-primary shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-foreground truncate">{file.filename}</p>
                          <p className="text-xs text-muted-foreground">
                            {(file.size / 1024).toFixed(2)} KB • Uploaded{" "}
                            {new Date(file.modified).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDeleteFile(file.filename)}
                        disabled={deletingFile === file.filename}
                        className="text-destructive hover:text-destructive hover:bg-destructive/10 gap-2 shrink-0"
                      >
                        {deletingFile === file.filename ? (
                          <>
                            <div className="h-4 w-4 animate-spin rounded-full border-2 border-destructive border-t-transparent" />
                            Deleting...
                          </>
                        ) : (
                          <>
                            <Trash2 className="h-4 w-4" />
                            Delete
                          </>
                        )}
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </section>
      )}

      {/* Main Dashboard */}
      <section className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <StaggerContainer className="grid gap-6 lg:grid-cols-3">
          {/* Controls Panel */}
          <StaggerItem className="lg:col-span-1">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-primary" />
                  Configuration
                </CardTitle>
                <CardDescription>Set up your training parameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* File Upload */}
                <div className="space-y-2">
                  <Label htmlFor="file-upload">Upload CSV Data</Label>
                  <div className="relative">
                    <input
                      id="file-upload"
                      type="file"
                      accept=".csv"
                      multiple
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <label
                      htmlFor="file-upload"
                      className="flex h-32 w-full cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-muted/30 transition-colors hover:border-primary hover:bg-muted/50"
                    >
                      <Upload className="h-8 w-8 text-muted-foreground mb-2" />
                      <span className="text-sm text-muted-foreground">
                        Click to upload CSV
                      </span>
                      <span className="text-xs text-muted-foreground mt-1">or drag and drop</span>
                    </label>
                  </div>

                  {/* File Preview Cards - Shows all selected files */}
                  {files.length > 0 && (
                    <div className="space-y-2">
                      {files.map((file, index) => (
                        <div
                          key={`${file.name}-${index}`}
                          className="rounded-lg border border-accent bg-accent/5 p-4 space-y-2 animate-in fade-in slide-in-from-top-2"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex items-start gap-3 flex-1 min-w-0">
                              <CheckCircle2 className="h-5 w-5 text-accent shrink-0 mt-0.5" />
                              <div className="flex-1 min-w-0">
                                <p className="font-semibold text-foreground text-sm truncate">{file.name}</p>
                                <p className="text-xs text-muted-foreground mt-1">
                                  {(file.size / 1024).toFixed(2)} KB
                                </p>
                              </div>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleRemoveFile(index)}
                              className="text-destructive hover:text-destructive hover:bg-destructive/10 shrink-0"
                            >
                              <Trash2 className="h-4 w-4" />
                              <span className="sr-only">Delete file</span>
                            </Button>
                          </div>
                        </div>
                      ))}
                      <p className="text-xs text-accent/80">
                        {files.length} file{files.length > 1 ? "s" : ""} ready to upload. Click "Start Forecasting" below to begin.
                      </p>
                    </div>
                  )}
                </div>

                {/* Training Model Selection */}
                <div className="space-y-2">
                  <Label htmlFor="model">Training Model</Label>
                  <Select value={trainingModel} onValueChange={setTrainingModel}>
                    <SelectTrigger id="model">
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                    <SelectItem value="GPT4TS">GPT4TS (Nonlinear)</SelectItem>
                    <SelectItem value="GPT4TS_LINEAR">GPT4TS (Linear)</SelectItem>
                    <SelectItem value="LLAMA">LLaMA (Nonlinear)</SelectItem>
                    <SelectItem value="LLAMA_LINEAR">LLaMA (Linear)</SelectItem>
                    <SelectItem value="BERT">BERT (Nonlinear)</SelectItem>
                    <SelectItem value="BERT_LINEAR">BERT (Linear)</SelectItem>
                    <SelectItem value="BART">BART (Nonlinear)</SelectItem>
                    <SelectItem value="BART_LINEAR">BART (Linear)</SelectItem>
                    <SelectItem value="OPT">OPT (Nonlinear)</SelectItem>
                    <SelectItem value="OPT_LINEAR">OPT (Linear)</SelectItem>
                    <SelectItem value="GEMMA">Gemma (Nonlinear)</SelectItem>
                    <SelectItem value="GEMMA_LINEAR">Gemma (Linear)</SelectItem>
                    <SelectItem value="QWEN">Qwen (Nonlinear)</SelectItem>
                    <SelectItem value="QWEN_LINEAR">Qwen (Linear)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Selected: {trainingModel}</p>
                </div>

                {/* Prediction Length */}
                <div className="space-y-2">
                  <Label htmlFor="prediction-length">Prediction Length (Steps)</Label>
                  <Select value={predictionLength} onValueChange={setPredictionLength}>
                    <SelectTrigger id="prediction-length">
                      <SelectValue placeholder="Select prediction length" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1 step</SelectItem>
                      <SelectItem value="3">3 steps</SelectItem>
                      <SelectItem value="6">6 steps</SelectItem>
                      <SelectItem value="36">36 steps</SelectItem>
                      <SelectItem value="72">72 steps</SelectItem>
                      <SelectItem value="144">144 steps</SelectItem>
                      <SelectItem value="432">432 steps</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Forecast {predictionLength} hours ahead</p>
                </div>

                {/* Dropout Rate */}
                <div className="space-y-2">
                  <Label htmlFor="dropout">Dropout Rate</Label>
                  <div className="flex gap-2 items-center">
                    <input
                      id="dropout"
                      type="number"
                      min="0"
                      max="0.5"
                      step="0.05"
                      value={dropoutRate}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value)
                        if (val >= 0 && val <= 0.5) {
                          setDropoutRate(e.target.value)
                        }
                      }}
                      className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                      placeholder="0.2"
                    />
                    <span className="text-sm text-muted-foreground font-mono">{dropoutRate}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">Regularization: 0.0 - 0.5</p>
                </div>

                {/* Training Mode */}
                <div className="space-y-2">
                  <Label htmlFor="mode">Training Mode</Label>
                  <Select value={mode} onValueChange={setMode}>
                    <SelectTrigger id="mode">
                      <SelectValue placeholder="Select mode" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="federated">
                        <div className="flex items-center gap-2">
                          <Shield className="h-4 w-4" />
                          Federated (Privacy-Preserving)
                        </div>
                      </SelectItem>
                      <SelectItem value="centralized">
                        <div className="flex items-center gap-2">
                          <Server className="h-4 w-4" />
                          Centralized (Server Upload)
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <div
                    className={`text-xs p-2 rounded ${mode === "federated" ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-700"}`}
                  >
                    {mode === "federated"
                      ? "Data stays on your device. Only model updates are shared."
                      : "Data is uploaded to server for training."}
                  </div>
                </div>

                {/* Federated Algorithm Selection - Only for Federated Mode */}
                {mode === "federated" && (
                  <div className="space-y-2 pt-2 border-t border-border">
                    <Label htmlFor="algorithm">Federated Learning Algorithm</Label>
                    <Select value={federalAlgorithm} onValueChange={setFederalAlgorithm}>
                      <SelectTrigger id="algorithm">
                        <SelectValue placeholder="Select algorithm" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="FedAvg">FedAvg - Federated Averaging</SelectItem>
                        <SelectItem value="FedProx">FedProx - Proximal Term</SelectItem>
                        <SelectItem value="SCAFFOLD">SCAFFOLD - Control Variates</SelectItem>
                        <SelectItem value="StatAvg">StatAvg - Statistics Averaging</SelectItem>
                        <SelectItem value="FedPer">FedPer - Personalized Head</SelectItem>
                        <SelectItem value="FedLN">FedLN - Local Layer Norms</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">Selected: {federalAlgorithm}</p>
                  </div>
                )}

                {/* Federated-only parameters */}
                {mode === "federated" && (
                  <>
                    {/* Number of Clients */}
                    <div className="space-y-2">
                      <Label htmlFor="clients">Number of Clients</Label>
                      <div className="flex gap-2 items-center">
                        <input
                          id="clients"
                          type="number"
                          min="1"
                          max="10"
                          value={numClients}
                          onChange={(e) => {
                            const val = parseInt(e.target.value)
                            if (val >= 1 && val <= 10) setNumClients(e.target.value)
                          }}
                          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                          placeholder="5"
                        />
                        <span className="text-sm text-muted-foreground font-mono">/ 10</span>
                      </div>
                      <p className="text-xs text-muted-foreground">Participating clients: 1 - 10</p>
                    </div>

                    {/* Number of Rounds */}
                    <div className="space-y-2">
                      <Label htmlFor="rounds">Communication Rounds</Label>
                      <div className="flex gap-2 items-center">
                        <input
                          id="rounds"
                          type="number"
                          min="1"
                          max="50"
                          value={numRounds}
                          onChange={(e) => {
                            const val = parseInt(e.target.value)
                            if (val >= 1 && val <= 50) setNumRounds(e.target.value)
                          }}
                          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                          placeholder="5"
                        />
                        <span className="text-sm text-muted-foreground font-mono">/ 50</span>
                      </div>
                      <p className="text-xs text-muted-foreground">FL aggregation rounds</p>
                    </div>

                    {/* Local Epochs */}
                    <div className="space-y-2">
                      <Label htmlFor="localEpochs">Local Epochs per Round</Label>
                      <div className="flex gap-2 items-center">
                        <input
                          id="localEpochs"
                          type="number"
                          min="1"
                          max="10"
                          value={localEpochs}
                          onChange={(e) => {
                            const val = parseInt(e.target.value)
                            if (val >= 1 && val <= 10) setLocalEpochs(e.target.value)
                          }}
                          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                          placeholder="1"
                        />
                        <span className="text-sm text-muted-foreground font-mono">/ 10</span>
                      </div>
                      <p className="text-xs text-muted-foreground">Training epochs per client per round</p>
                    </div>

                    {/* LLM Layers */}
                    <div className="space-y-2">
                      <Label htmlFor="llmLayers">LLM Layers</Label>
                      <div className="flex gap-2 items-center">
                        <input
                          id="llmLayers"
                          type="number"
                          min="1"
                          max="12"
                          value={llmLayers}
                          onChange={(e) => {
                            const val = parseInt(e.target.value)
                            if (val >= 1 && val <= 12) setLlmLayers(e.target.value)
                          }}
                          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                          placeholder="4"
                        />
                        <span className="text-sm text-muted-foreground font-mono">/ 12</span>
                      </div>
                      <p className="text-xs text-muted-foreground">Transformer layers used from the LLM backbone</p>
                    </div>
                  </>
                )}

                {/* Start Button */}
                <Button
                  className="w-full gap-2"
                  onClick={handleStartTraining}
                  disabled={isProcessing || files.length === 0}
                >
                  {isProcessing ? (
                    <>
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4" />
                      Start Forecasting
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </StaggerItem>

          {/* Results Panel */}
          <StaggerItem className="lg:col-span-2">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-primary" />
                  Wind Speed Forecasting Results
                </CardTitle>
                <CardDescription>
                  {showResults && trainingResult
                    ? `${trainingResult.prediction_length}-step wind speed forecast using ${trainingResult.model_name} (${mode === "federated" ? "Federated" : "Centralized"})`
                    : showResults
                      ? `${predictionLength}-step forecast ${mode === "federated" ? "(Federated Model)" : "(Centralized Model)"}`
                      : "Configure and run to see predictions"}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {showResults ? (
                  <div className="space-y-6">
                    {/* Stats Row - Only MAE + RMSE + Time */}
                    <div className="grid gap-4 sm:grid-cols-3">
                      <div className="rounded-lg border border-border bg-muted/30 p-4">
                        <div className="flex items-center gap-2 text-muted-foreground text-sm">
                          <Wind className="h-4 w-4" />
                          MAE
                        </div>
                        <div className="mt-1 text-2xl font-bold text-foreground">
                          {trainingResult ? trainingResult.metrics.mae : "--"}
                        </div>
                        <div className="text-xs text-muted-foreground">Mean Absolute Error (normalized)</div>
                      </div>
                      <div className="rounded-lg border border-border bg-muted/30 p-4">
                        <div className="flex items-center gap-2 text-muted-foreground text-sm">
                          <BarChart3 className="h-4 w-4" />
                          RMSE
                        </div>
                        <div className="mt-1 text-2xl font-bold text-foreground">
                          {trainingResult ? trainingResult.metrics.rmse : "--"}
                        </div>
                        <div className="text-xs text-muted-foreground">Root Mean Square Error (normalized)</div>
                      </div>
                      <div className="rounded-lg border border-border bg-muted/30 p-4">
                        <div className="flex items-center gap-2 text-muted-foreground text-sm">
                          <Clock className="h-4 w-4" />
                          Time
                        </div>
                        <div className="mt-1 text-2xl font-bold text-foreground">
                          {trainingResult ? `${trainingResult.training_time_seconds}s` : "--s"}
                        </div>
                        <div className="text-xs text-muted-foreground">Training Duration</div>
                      </div>
                    </div>

                    {/* Federated Mode Info Panel */}
                    {mode === "federated" && trainingResult && (
                      <div className="rounded-lg border border-green-200 bg-green-50 p-4">
                        <h4 className="font-semibold text-green-800 flex items-center gap-2 mb-3">
                          <Shield className="h-4 w-4 text-green-600" />
                          Federated Learning - Privacy Preserved
                        </h4>
                        <div className="grid gap-3 sm:grid-cols-3">
                          <div>
                            <div className="text-xs text-muted-foreground">Algorithm</div>
                            <div className="text-lg font-semibold text-foreground">
                              {federalAlgorithm}
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-muted-foreground">Clients</div>
                            <div className="text-lg font-semibold text-foreground">
                              {numClients}
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-muted-foreground">Model</div>
                            <div className="text-lg font-semibold text-foreground">
                              {trainingResult.model_name}
                            </div>
                          </div>
                        </div>
                        <p className="text-xs text-green-600 mt-3">
                          Data stayed on each client device. Only model updates were aggregated on the server.
                        </p>
                      </div>
                    )}

                    {/* Wind Speed Chart */}
                    <div>
                      <h4 className="font-semibold text-foreground mb-3 flex items-center gap-2">
                        <Wind className="h-4 w-4 text-primary" />
                        Wind Speed Forecast
                        {trainingResult && trainingResult.forecast.length > 0 && (
                          <span className="text-xs font-normal text-muted-foreground ml-1">
                            — {trainingResult.forecast.length} steps (normalized scale)
                          </span>
                        )}
                      </h4>

                      <div style={{ width: "100%", height: 280 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={
                              trainingResult && trainingResult.forecast.length > 0
                                ? trainingResult.forecast.map((p) => ({
                                    label: `S${p.step}`,
                                    predicted: p.predicted,
                                    ...(p.actual !== null && p.actual !== undefined
                                      ? { actual: p.actual }
                                      : {}),
                                  }))
                                : mockForecastData.map((d) => ({
                                    label: d.hour,
                                    predicted: d.predicted,
                                    actual: d.actual,
                                  }))
                            }
                            margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                            <XAxis
                              dataKey="label"
                              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                              interval="preserveStartEnd"
                            />
                            <YAxis
                              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                              label={{
                                value: trainingResult && trainingResult.forecast.length > 0 ? "normalized" : "m/s",
                                angle: -90,
                                position: "insideLeft",
                                fontSize: 12,
                                fill: "hsl(var(--muted-foreground))",
                              }}
                              domain={["auto", "auto"]}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "hsl(var(--background))",
                                border: "1px solid hsl(var(--border))",
                                borderRadius: "6px",
                                fontSize: "12px",
                                color: "hsl(var(--foreground))",
                              }}
                              formatter={(value: number, name: string) => [
                                trainingResult && trainingResult.forecast.length > 0
                                  ? Number(value).toFixed(4)
                                  : `${Number(value).toFixed(2)} m/s`,
                                name === "predicted" ? "Predicted" : "Actual",
                              ]}
                              labelFormatter={(label) => `${label}`}
                            />
                            <Legend wrapperStyle={{ fontSize: "12px" }} />
                            {(!trainingResult ||
                              trainingResult.forecast.some(
                                (p) => p.actual !== null && p.actual !== undefined,
                              )) && (
                              <Line
                                type="monotone"
                                dataKey="actual"
                                stroke="#A23B72"
                                strokeWidth={2}
                                dot={false}
                                name="Actual"
                                connectNulls={false}
                                isAnimationActive={false}
                              />
                            )}
                            <Line
                              type="monotone"
                              dataKey="predicted"
                              stroke="#2E86AB"
                              strokeWidth={2}
                              strokeDasharray="5 5"
                              dot={false}
                              name="Predicted"
                              isAnimationActive={false}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Download Results */}
                    {trainingResult && (
                      trainingResult.download_training_summary ||
                      trainingResult.download_timing_summary
                    ) && (
                      <div className="rounded-lg border border-border bg-muted/20 p-4">
                        <h4 className="font-semibold text-foreground text-sm mb-3 flex items-center gap-2">
                          <FileText className="h-4 w-4 text-primary" />
                          Download Training Results
                        </h4>
                        <div className="flex flex-wrap gap-3">
                          {trainingResult?.download_training_summary && (
                            <a
                              href={`http://localhost:8001${trainingResult.download_training_summary}`}
                              download="training_summary.csv"
                              className="inline-flex items-center gap-2 rounded-md border border-border bg-background px-4 py-2 text-sm font-medium text-foreground shadow-sm hover:bg-muted/50 transition-colors"
                            >
                              {/* inline SVG so no extra import needed */}
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                className="h-4 w-4 text-primary"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth={2}
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              >
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="7 10 12 15 17 10" />
                                <line x1="12" y1="15" x2="12" y2="3" />
                              </svg>
                              training_summary.csv
                            </a>
                          )}
                          {trainingResult?.download_timing_summary && (
                            <a
                              href={`http://localhost:8001${trainingResult.download_timing_summary}`}
                              download="timing_summary.csv"
                              className="inline-flex items-center gap-2 rounded-md border border-border bg-background px-4 py-2 text-sm font-medium text-foreground shadow-sm hover:bg-muted/50 transition-colors"
                            >
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                className="h-4 w-4 text-primary"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth={2}
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              >
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="7 10 12 15 17 10" />
                                <line x1="12" y1="15" x2="12" y2="3" />
                              </svg>
                              timing_summary.csv
                            </a>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground mt-2">
                          Per-round metrics and wall-clock timing from this training run.
                        </p>
                      </div>
                    )}

                    {/* Training Info (if complete) */}
                    {trainingResult && (
                      <div className="rounded-lg border border-border bg-muted/20 p-4">
                        <h4 className="font-semibold text-foreground text-sm mb-2">Training Summary</h4>
                        <div className="grid gap-2 sm:grid-cols-4 text-sm">
                          <div>
                            <span className="text-muted-foreground">Model:</span>{" "}
                            <span className="text-foreground">{trainingResult.model_name}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Horizon:</span>{" "}
                            <span className="text-foreground">{trainingResult.prediction_length} steps</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Dropout:</span>{" "}
                            <span className="text-foreground">{trainingResult.dropout_rate}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Time:</span>{" "}
                            <span className="text-foreground">{trainingResult.training_time_seconds}s</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex h-[400px] flex-col items-center justify-center text-center">
                    <Clock className="h-12 w-12 text-muted-foreground mb-4" />
                    <h4 className="font-semibold text-foreground">No Results Yet</h4>
                    <p className="text-sm text-muted-foreground mt-1 max-w-xs">
                      Upload your data and click &quot;Start Forecasting&quot; to see wind speed predictions.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </StaggerItem>
        </StaggerContainer>
      </section>

      {/* Feedback + Recent Answers */}
      <section className="mx-auto max-w-7xl px-4 pb-12 sm:px-6 lg:px-8">
        <div className="space-y-6">
          <div className="rounded-lg border border-border bg-card p-6">
            <div className="flex items-center gap-2 mb-6">
              <MessageSquare className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold text-foreground">Ask Questions or Send Us Feedback</h3>
              <span className="text-xs text-muted-foreground ml-auto">
                
              </span>
            </div>

            <p className="text-sm text-muted-foreground mb-4">
              Answered questions may appear anonymously in the Recent Answers section below.
            </p>

            {feedbackSent ? (
              <div className="flex items-center gap-3 rounded-lg border border-green-200 bg-green-50 p-4">
                <CheckCircle2 className="h-5 w-5 text-green-600 shrink-0" />
                <div>
                  <p className="font-medium text-green-900">Thank you for your entry!</p>
                  <p className="text-sm text-green-700">We appreciate your input and will review it shortly.</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">
                    Your Name (optional)
                  </label>
                  <input
                    type="text"
                    value={feedbackName}
                    onChange={(e) => setFeedbackName(e.target.value)}
                    placeholder="Enter your name or leave blank for anonymous"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium text-foreground block mb-2">
                    Your Input
                  </label>
                  <textarea
                    value={feedbackMessage}
                    onChange={(e) => setFeedbackMessage(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && e.ctrlKey && !sendingFeedback && feedbackMessage.trim()) {
                        handleSubmitFeedback()
                      }
                    }}
                    placeholder="Share your thoughts, suggestions, or report issues... (Ctrl+Enter to submit)"
                    className="w-full rounded-md border border-input bg-background px-3 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 resize-vertical min-h-24"
                  />
                  <p className="text-xs text-muted-foreground mt-1.5">
                    {feedbackMessage.length} characters • Ctrl+Enter to submit
                  </p>
                </div>

                <div className="flex justify-end gap-2">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setFeedbackMessage("")
                      setFeedbackName("")
                    }}
                    disabled={sendingFeedback}
                    className="px-4"
                  >
                    Clear
                  </Button>
                  <Button
                    onClick={handleSubmitFeedback}
                    disabled={sendingFeedback || !feedbackMessage.trim()}
                    className="gap-2 px-4"
                  >
                    {sendingFeedback ? (
                      <>
                        <div className="h-4 w-4 animate-spin rounded-full border-2 border-background border-t-transparent" />
                        Sending...
                      </>
                    ) : (
                      <>
                        <Send className="h-4 w-4" />
                        Send Feedback
                      </>
                    )}
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div className="rounded-lg border border-border bg-card p-6">
            <div className="flex items-center gap-2 mb-6">
              <CheckCircle2 className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold text-foreground">Recent Answers</h3>

              <Button
                variant="outline"
                size="sm"
                onClick={handleRefreshAnswers}
                disabled={loadingPublicAnswers || !apiConnected}
                className="ml-auto gap-2 bg-transparent"
              >
                <RefreshCw className={`h-4 w-4 ${loadingPublicAnswers ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>

            {loadingPublicAnswers ? (
              <div className="flex items-center justify-center py-8 text-muted-foreground">
                <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent mr-3" />
                Loading recent answers...
              </div>
            ) : publicAnswers.length === 0 ? (
              <div className="rounded-lg border border-dashed border-border bg-muted/20 p-6 text-center">
                <MessageSquare className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
                <h4 className="font-semibold text-foreground">No public answers yet</h4>
                <p className="text-sm text-muted-foreground mt-1">
                  Answered questions will appear here so everyone can see them.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {visiblePublicAnswers.map((item) => (
                  <div
                    key={item.id}
                    className="rounded-lg border border-border bg-muted/20 p-4"
                  >
                    <div
                      id={
                        getThreadRootQuestionMessage(item.id)
                          ? getMessageElementId(getThreadRootQuestionMessage(item.id)!.id)
                          : undefined
                      }
                    >
                      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground mb-1">
                        Question
                      </p>
                      <p className="text-foreground font-medium">{item.question}</p>
                    </div>

                    <div
                      id={
                        getMainOfficialAnswerMessage(item.id)
                          ? getMessageElementId(getMainOfficialAnswerMessage(item.id)!.id)
                          : undefined
                      }
                      className="mt-4 rounded-md border border-primary/10 bg-primary/5 p-4"
                    >
                      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground mb-1">
                        Official Answer
                      </p>
                      <p className="text-sm text-foreground whitespace-pre-wrap">
                        {item.answer_text}
                      </p>
                    </div>

                    <div className="mt-3 text-xs text-muted-foreground">
                      Answered on {new Date(item.answered_at).toLocaleString()}
                    </div>

                    <div className="mt-4 flex flex-wrap gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => toggleConversationThread(item.id)}
                      >
                        {expandedConversationId === item.id ? "Hide Thread" : "View Thread"}
                      </Button>

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openReplyFormForQuestion(item.id)}
                      >
                        {replyConversationId === item.id &&
                        replyToMessageId === getThreadRootQuestionMessage(item.id)?.id
                          ? "Cancel Reply"
                          : "Reply to Question"}
                      </Button>
                    </div>

                    {replyConversationId === item.id &&
                      replyToMessageId === getThreadRootQuestionMessage(item.id)?.id &&
                      renderReplyForm(item.id, getThreadRootQuestionMessage(item.id)!.id)}

                    {expandedConversationId === item.id && (
                      <div className="mt-4 rounded-lg border border-border bg-background p-4">
                        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground mb-3">
                          Conversation Thread
                        </p>

                        {loadingConversationId === item.id ? (
                          <div className="flex items-center text-sm text-muted-foreground">
                            <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent mr-2" />
                            Loading thread...
                          </div>
                        ) : (
                          <div className="space-y-4">
                            {getTopLevelThreadMessages(item.id).length === 0 &&
                            getOrphanPublicReplies(item.id).length === 0 ? (
                              <p className="text-sm text-muted-foreground">
                                No additional follow-ups yet.
                              </p>
                            ) : (
                              <>
                                {getTopLevelThreadMessages(item.id).map((message) =>
                                  renderConversationNode(item.id, message),
                                )}

                                {getOrphanPublicReplies(item.id).length > 0 && (
                                  <div className="space-y-2">
                                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                                      Additional Replies
                                    </p>

                                    {getOrphanPublicReplies(item.id).map((reply) => (
                                      <div
                                        key={reply.id}
                                        id={getMessageElementId(reply.id)}
                                        className="rounded-md border border-primary/20 bg-primary/5 p-3"
                                      >
                                        <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                                          <span>
                                            Official Reply • {new Date(reply.created_at).toLocaleString()}
                                          </span>

                                          {reply.reply_to_message_id && (
                                            <button
                                              type="button"
                                              onClick={() => scrollToMessage(reply.reply_to_message_id)}
                                              className="italic underline underline-offset-2 hover:text-foreground"
                                            >
                                              Re: {getReplyPreviewText(item.id, reply.reply_to_message_id)}
                                            </button>
                                          )}
                                        </div>

                                        <p className="text-sm text-foreground whitespace-pre-wrap">
                                          {reply.message_text}
                                        </p>

                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          onClick={() => openReplyFormForMessage(item.id, reply.id)}
                                          className="mt-2 h-auto p-0 text-xs font-bold text-muted-foreground hover:bg-transparent hover:text-foreground"
                                        >
                                          Reply
                                        </Button>

                                        {renderReplyForm(item.id, reply.id)}
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    )}


                  </div>
                ))}

                {publicAnswers.length > ANSWERS_PER_PAGE && (
                  <div className="flex flex-col gap-3 pt-2 sm:flex-row sm:items-center sm:justify-between">
                    <p className="text-sm text-muted-foreground">
                      Showing {startAnswerIndex + 1}-
                      {Math.min(endAnswerIndex, publicAnswers.length)} of {publicAnswers.length} answers
                    </p>

                    <div className="flex flex-wrap items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentAnswersPage(1)}
                        disabled={currentAnswersPage === 1}
                      >
                        First
                      </Button>

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentAnswersPage((prev) => Math.max(prev - 1, 1))}
                        disabled={currentAnswersPage === 1}
                      >
                        Previous
                      </Button>

                      {answerPageNumbers.map((pageNumber) => (
                        <Button
                          key={pageNumber}
                          variant={currentAnswersPage === pageNumber ? "default" : "outline"}
                          size="sm"
                          onClick={() => setCurrentAnswersPage(pageNumber)}
                          className="min-w-10"
                        >
                          {pageNumber}
                        </Button>
                      ))}

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() =>
                          setCurrentAnswersPage((prev) => Math.min(prev + 1, totalAnswerPages))
                        }
                        disabled={currentAnswersPage === totalAnswerPages}
                      >
                        Next
                      </Button>

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentAnswersPage(totalAnswerPages)}
                        disabled={currentAnswersPage === totalAnswerPages}
                      >
                        Last
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>
    </PageLayout>
  )
}
