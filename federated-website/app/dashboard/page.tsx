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
  type HealthResponse,
  type TrainingResult,
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
} from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

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
        "dashboard"
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
          // Load files if API is connected
          loadUploadedFiles()
        }
      } catch {
        setApiConnected(false)
      }
    }
    checkApiConnection()
  }, [])

  // File upload handler
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files)
      setFiles(prev => [...prev, ...newFiles])
      setError(null)
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
        }

        const result = await startTraining(trainFilename, config)
        setTrainingResult(result)
        setShowResults(true)

        // Reset file selection
        setFiles([])
        const fileInput = document.getElementById("file-upload") as HTMLInputElement
        if (fileInput) fileInput.value = ""
      } else {
        // Federated mode: Upload files -> then run federated training
        if (files.length === 0) {
          setError("Please select at least one CSV file to upload")
          return
        }

        // Step 1: Upload all files
        const savedFilenames: string[] = []
        for (let i = 0; i < files.length; i++) {
          const file = files[i]
          setProcessingStep(`Uploading file ${i + 1} of ${files.length}: ${file.name}...`)
          const uploadResult = await uploadFile(file)
          savedFilenames.push(uploadResult.file.filename)
        }

        // Reload file list after uploads
        await loadUploadedFiles()

        // Step 2: Run federated training via /api/train with mode=federated
        const trainFilename = savedFilenames[savedFilenames.length - 1]
        setProcessingStep(
          `Federated training: ${trainingModel} / ${federalAlgorithm} / ${numClients} clients...`
        )

        const config = {
          training_model: trainingModel,
          prediction_length: parseInt(predictionLength),
          dropout_rate: parseFloat(dropoutRate),
          mode: "federated" as const,
          federated_algorithm: federalAlgorithm,
          num_clients: parseInt(numClients),
        }

        const result = await startTraining(trainFilename, config)
        setTrainingResult(result)
        setShowResults(true)

        // Reset file selection
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

      {/* Training Success (Centralized Mode) */}
      {trainingResult && (
        <section className="mx-auto max-w-7xl px-4 pt-4 sm:px-6 lg:px-8">
          <div className="rounded-lg border border-green-200 bg-green-50 p-4 flex items-start gap-3">
            <CheckCircle2 className="h-5 w-5 text-green-600 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-green-800">Training Complete</h4>
              <p className="text-sm text-green-700">
                {trainingResult.message} | MAE: {trainingResult.metrics.mae} | RMSE: {trainingResult.metrics.rmse}
                {trainingResult.metrics.mape != null && ` | MAPE: ${trainingResult.metrics.mape}%`}
                {" | Time: "}{trainingResult.training_time_seconds}s
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
                      <SelectItem value="GPT4TS">GPT4TS - GPT-based Time Series</SelectItem>
                      <SelectItem value="LLAMA">LLAMA - LLaMA Language Model</SelectItem>
                      <SelectItem value="BERT">BERT - Bidirectional Encoder</SelectItem>
                      <SelectItem value="BART">BART - Denoising Autoencoder</SelectItem>
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
                        <SelectItem value="FedBN">FedBN - Batch Normalization</SelectItem>
                        <SelectItem value="FedPer">FedPer - Personalization</SelectItem>
                        <SelectItem value="SCAFFOLD">SCAFFOLD - Control Variates</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">Selected: {federalAlgorithm}</p>
                  </div>
                )}

                {/* Number of Clients - Only for Federated Mode */}
                {mode === "federated" && (
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
                          if (val >= 1 && val <= 10) {
                            setNumClients(e.target.value)
                          }
                        }}
                        className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
                        placeholder="5"
                      />
                      <span className="text-sm text-muted-foreground font-mono">/ 10</span>
                    </div>
                    <p className="text-xs text-muted-foreground">Participating clients: 1 - 10</p>
                  </div>
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
                    {/* Stats Row - Real Metrics from Training */}
                    <div className="grid gap-4 sm:grid-cols-3">
                      <div className="rounded-lg border border-border bg-muted/30 p-4">
                        <div className="flex items-center gap-2 text-muted-foreground text-sm">
                          <Wind className="h-4 w-4" />
                          MAE
                        </div>
                        <div className="mt-1 text-2xl font-bold text-foreground">
                          {trainingResult ? `${trainingResult.metrics.mae} m/s` : "-- m/s"}
                        </div>
                        <div className="text-xs text-muted-foreground">Mean Absolute Error</div>
                      </div>
                      <div className="rounded-lg border border-border bg-muted/30 p-4">
                        <div className="flex items-center gap-2 text-muted-foreground text-sm">
                          <TrendingUp className="h-4 w-4" />
                          MAPE
                        </div>
                        <div className="mt-1 text-2xl font-bold text-foreground">
                          {trainingResult?.metrics.mape != null ? `${trainingResult.metrics.mape}%` : "-- %"}
                        </div>
                        <div className="text-xs text-muted-foreground">Mean Absolute % Error</div>
                      </div>
                      <div className="rounded-lg border border-border bg-muted/30 p-4">
                        <div className="flex items-center gap-2 text-muted-foreground text-sm">
                          <BarChart3 className="h-4 w-4" />
                          RMSE
                        </div>
                        <div className="mt-1 text-2xl font-bold text-foreground">
                          {trainingResult ? `${trainingResult.metrics.rmse} m/s` : "-- m/s"}
                        </div>
                        <div className="text-xs text-muted-foreground">Root Mean Square Error</div>
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
                      </h4>
                      <ChartContainer
                        config={{
                          actual: {
                            label: "Actual",
                            color: "hsl(var(--chart-1))",
                          },
                          predicted: {
                            label: "Predicted",
                            color: "hsl(var(--chart-2))",
                          },
                        }}
                        className="h-[280px] w-full"
                      >
                        <LineChart data={
                          trainingResult
                            ? trainingResult.forecast.map((p) => ({
                                hour: `Step ${p.step}`,
                                actual: p.actual,
                                predicted: p.predicted,
                              }))
                            : mockForecastData
                        }>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                          <XAxis dataKey="hour" tick={{ fontSize: 12 }} className="text-muted-foreground" />
                          <YAxis
                            tick={{ fontSize: 12 }}
                            className="text-muted-foreground"
                            label={{ value: "m/s", angle: -90, position: "insideLeft", fontSize: 12 }}
                          />
                          <ChartTooltip content={<ChartTooltipContent />} />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="actual"
                            stroke="var(--color-actual)"
                            strokeWidth={2}
                            dot={{ r: 3 }}
                            name="Actual"
                          />
                          <Line
                            type="monotone"
                            dataKey="predicted"
                            stroke="var(--color-predicted)"
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            dot={{ r: 3 }}
                            name="Predicted"
                          />
                        </LineChart>
                      </ChartContainer>
                    </div>

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

      {/* Feedback Section - Expanded Card */}
      <section className="mx-auto max-w-7xl px-4 pb-12 sm:px-6 lg:px-8">
        <div className="rounded-lg border border-border bg-card p-6">
          <div className="flex items-center gap-2 mb-6">
            <MessageSquare className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-semibold text-foreground">Send Us Feedback</h3>
            <span className="text-xs text-muted-foreground ml-auto">Visible to creators only</span>
          </div>

          {feedbackSent ? (
            <div className="flex items-center gap-3 rounded-lg border border-green-200 bg-green-50 p-4">
              <CheckCircle2 className="h-5 w-5 text-green-600 shrink-0" />
              <div>
                <p className="font-medium text-green-900">Thank you for your feedback!</p>
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
                  Your Feedback
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
      </section>
    </PageLayout>
  )
}
