"use client"

import React from "react"
import { TrendingUp } from "lucide-react"

import { PageLayout, StaggerContainer, StaggerItem } from "@/components/page-layout"
import { SectionHeader } from "@/components/section-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Wind,
  Server,
  ArrowRight,
  ArrowDown,
  RefreshCw,
  Layers,
  Database,
  Cpu,
  Building2,
  Globe,
  CheckCircle2,
} from "lucide-react"

// Component for visualizing clients/wind farms
function ClientNode({ name, id }: { name: string; id: string }) {
  return (
    <div className="flex flex-col items-center">
      <div className="relative">
        <div className="flex h-16 w-16 items-center justify-center rounded-xl border-2 border-primary/30 bg-card shadow-sm">
          <Building2 className="h-7 w-7 text-primary" />
        </div>
        <div className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-accent text-[10px] font-bold text-accent-foreground">
          {id}
        </div>
      </div>
      <span className="mt-2 text-xs font-medium text-muted-foreground">{name}</span>
    </div>
  )
}

// Animated arrow component
function FlowArrow({ direction = "down", label }: { direction?: "down" | "right" | "up"; label?: string }) {
  const Icon = direction === "right" ? ArrowRight : ArrowDown
  const rotateClass = direction === "up" ? "rotate-180" : ""
  
  return (
    <div className={`flex flex-col items-center ${direction === "right" ? "flex-row" : ""}`}>
      <div className={`p-2 ${rotateClass}`}>
        <Icon className="h-5 w-5 text-primary animate-pulse" />
      </div>
      {label && (
        <span className="text-[10px] text-muted-foreground font-medium whitespace-nowrap">{label}</span>
      )}
    </div>
  )
}

export default function ArchitecturePage() {
  return (
    <PageLayout>
      {/* Header */}
      <section className="bg-gradient-to-b from-primary/5 to-background">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 sm:py-20 lg:px-8">
          <div className="mx-auto max-w-3xl text-center">
            <SectionHeader
              badge="Technical Overview"
              title="System Architecture"
              description=""
              centered
            />
          </div>
        </div>
      </section>

      {/* Main Architecture Diagram */}
      <section className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <Card className="overflow-hidden">
          <CardHeader className="bg-muted/50 border-b">
            <CardTitle className="flex items-center gap-2">
              <Globe className="h-5 w-5 text-primary" />
              Federated Learning Architecture
            </CardTitle>
            <CardDescription>
              Multiple wind farms collaborate to train a shared model without sharing raw data
            </CardDescription>
          </CardHeader>
          <CardContent className="p-8">
            <div className="flex flex-col items-center">
              {/* Clients Row */}
              <div className="flex flex-wrap items-center justify-center gap-6 sm:gap-10">
                <ClientNode name="Wind Farm A" id="1" />
                <ClientNode name="Wind Farm B" id="2" />
                <ClientNode name="Wind Farm C" id="3" />
                <ClientNode name="Region D" id="4" />
              </div>

              {/* Bidirectional arrows */}
              <div className="my-6 flex items-center gap-4">
                <div className="flex flex-col items-center">
                  <ArrowDown className="h-5 w-5 text-primary" />
                  <span className="text-[10px] text-muted-foreground mt-1">Model Updates</span>
                </div>
                <div className="w-8" />
                <div className="flex flex-col items-center">
                  <ArrowDown className="h-5 w-5 text-accent rotate-180" />
                  <span className="text-[10px] text-muted-foreground mt-1">Global Model</span>
                </div>
              </div>

              {/* Central Server */}
              <div className="relative">
                <div className="flex h-24 w-40 flex-col items-center justify-center rounded-2xl border-2 border-accent bg-gradient-to-br from-accent/10 to-accent/5 shadow-lg">
                  <Server className="h-8 w-8 text-accent" />
                  <span className="mt-1 text-sm font-semibold text-foreground">Central Server</span>
                </div>
                <div className="absolute -top-2 left-1/2 -translate-x-1/2 rounded-full bg-primary px-2 py-0.5 text-[10px] font-medium text-primary-foreground">
                  FedAvg
                </div>
              </div>

              {/* Output */}
              <FlowArrow direction="down" />

              <div className="flex h-16 w-48 items-center justify-center rounded-xl border border-border bg-card shadow-sm">
                <div className="text-center">
                  <Wind className="mx-auto h-5 w-5 text-primary" />
                  <span className="text-xs font-medium text-foreground">Aggregated Global Model</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>


      {/* Federated Learning Algorithms */}
      <section className="py-12 sm:py-16">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <SectionHeader
            badge="Algorithms"
            title="Federated Learning Algorithms"
            description=""
            centered
          />
        </div>
      </section>

      {/* FedAvg Detailed Explanation */}
      <section className="mx-auto max-w-7xl px-4 py-12 sm:px-6 sm:py-16 lg:px-8">
        <div className="grid gap-8 lg:grid-cols-2 items-start">
          <div>
            <SectionHeader
              badge="Core Algorithm"
              title="FedAvg: Federated Averaging (Default)"
              description="The foundational algorithm that powers privacy-preserving collaborative learning."
            />

            <div className="mt-6 space-y-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">Definition</h4>
                <p className="text-sm text-muted-foreground">
                  FedAvg computes a weighted average of model parameters from all participating clients based on their dataset sizes. 
                  This simple yet powerful approach enables global model training without centralizing data.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">How It Works</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">1.</span> Server sends current global model to all clients
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">2.</span> Each client trains on local data for multiple epochs
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">3.</span> Clients send weight updates back to server
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">4.</span> Server aggregates updates as: w_global = Σ(n_k / n) × w_k
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Why Use FedAvg</h4>
                <ul className="space-y-1.5 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Communication efficient with multiple local epochs
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Convergence guarantees even with heterogeneous data
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Simple implementation and easy to understand
                  </li>
                </ul>
              </div>

              {/* <div>
                <h4 className="font-semibold text-foreground mb-2">Wind Forecasting Application</h4>
                <p className="text-sm text-muted-foreground">
                  Ideal when wind farms have similar data distributions. Works well for regions with comparable weather patterns.
                </p>
              </div> */}
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">FedAvg Algorithm</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="rounded-lg bg-muted p-4">
                <code className="text-xs text-muted-foreground">
                  <span className="text-primary">// Federated Averaging</span>
                  <br />
                  <br />
                  <span className="text-foreground">for</span> each round r = 1, 2, ... :
                  <br />
                  &nbsp;&nbsp;<span className="text-foreground">for</span> each client k in parallel:
                  <br />
                  &nbsp;&nbsp;&nbsp;&nbsp;w_k ← LocalTrain(global_model, local_data_k)
                  <br />
                  <br />
                  &nbsp;&nbsp;<span className="text-accent">// Weighted average aggregation</span>
                  <br />
                  &nbsp;&nbsp;w_global ← Σ (n_k / n) × w_k
                  <br />
                  <br />
                  <span className="text-muted-foreground">// n_k = local dataset size</span>
                  <br />
                  <span className="text-muted-foreground">// n = total samples</span>
                </code>
              </div>
              <div>
                <h4 className="font-semibold text-xs text-foreground mb-2">Advantages</h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  <li>• Mathematically proven convergence</li>
                  <li>• Handles non-IID data reasonably well</li>
                  <li>• Minimal overhead for aggregation</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-xs text-foreground mb-2">Trade-offs</h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  <li>• Can struggle with highly heterogeneous data</li>
                  <li>• No built-in personalization capability</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* FedProx Detailed Explanation */}
      <section className="bg-muted/30 mx-auto max-w-7xl px-4 py-12 sm:px-6 sm:py-16 lg:px-8">
        <div className="grid gap-8 lg:grid-cols-2 items-start">
          <div>
            <SectionHeader
              badge="Algorithm"
              title="FedProx: Proximal Term Regularization"
              description="Addresses data heterogeneity and system heterogeneity in federated learning."
            />

            <div className="mt-6 space-y-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">Definition</h4>
                <p className="text-sm text-muted-foreground">
                  FedProx adds a proximal term (regularization penalty) to each client's local training objective. 
                  This term penalizes updates that drift too far from the global model, preventing divergence in heterogeneous settings.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">How It Works</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">1.</span> Modify local loss: L(w) + (μ/2)||w - w_global||²
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">2.</span> Each client minimizes their modified loss locally
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">3.</span> Server aggregates using weighted averaging (like FedAvg)
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">4.</span> The μ parameter (proximal coefficient) controls drift tolerance
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Why Use FedProx</h4>
                <ul className="space-y-1.5 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Handles non-IID data better than FedAvg
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Robust to device heterogeneity (slow/fast clients)
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Improves training stability in heterogeneous networks
                  </li>
                </ul>
              </div>

              {/* <div>
                <h4 className="font-semibold text-foreground mb-2">Wind Forecasting Application</h4>
                <p className="text-sm text-muted-foreground">
                  Recommended for multi-regional wind farms with different weather patterns. Handles disparate data distributions across regions.
                </p>
              </div> */}
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">FedProx Modification</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="rounded-lg bg-muted p-4">
                <code className="text-xs text-muted-foreground">
                  <span className="text-primary">// Local objective with proximal term</span>
                  <br />
                  <br />
                  <span className="text-foreground">minimize</span> L(w) + (μ/2)||w - w_t||²
                  <br />
                  <br />
                  <span className="text-primary">// Where:</span>
                  <br />
                  <span className="text-muted-foreground">// L(w) = original loss on local data</span>
                  <br />
                  <span className="text-muted-foreground">// w_t = global model at round t</span>
                  <br />
                  <span className="text-muted-foreground">// μ = proximal coefficient (typical: 0.01-0.1)</span>
                  <br />
                  <br />
                  <span className="text-accent">// Aggregation remains the same as FedAvg</span>
                </code>
              </div>
              <div>
                <h4 className="font-semibold text-xs text-foreground mb-2">Advantages</h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  <li>• Better convergence with heterogeneous data</li>
                  <li>• Tolerates system heterogeneity</li>
                  <li>• Backward compatible with FedAvg (μ=0)</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-xs text-foreground mb-2">Trade-offs</h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  <li>• Requires tuning μ parameter</li>
                  <li>• Slightly increased computation cost per client</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* FedBN Detailed Explanation */}
      <section className="mx-auto max-w-7xl px-4 py-12 sm:px-6 sm:py-16 lg:px-8">
        <div className="grid gap-8 lg:grid-cols-2 items-start">
          <div>
            <SectionHeader
              badge="Algorithm"
              title="FedBN: Federated Batch Normalization"
              description="Enables personalization by decoupling batch norm from global model aggregation."
            />

            <div className="mt-6 space-y-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">Definition</h4>
                <p className="text-sm text-muted-foreground">
                  FedBN keeps batch normalization parameters (mean, variance) local to each client while aggregating 
                  all other model weights. This enables personalization without maintaining separate models.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">How It Works</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">1.</span> Model split: BN params (local) + other weights (global)
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">2.</span> Server sends global model without BN statistics
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">3.</span> Each client keeps and updates own BN statistics during training
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">4.</span> Only non-BN weights aggregated (weighted average)
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Why Use FedBN</h4>
                <ul className="space-y-1.5 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Personalization without extra overhead
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Handles domain shift gracefully
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Improves accuracy on heterogeneous datasets
                  </li>
                </ul>
              </div>

              {/* <div>
                <h4 className="font-semibold text-foreground mb-2">Wind Forecasting Application</h4>
                <p className="text-sm text-muted-foreground">
                  Ideal for federated scenarios where each wind farm has local climate variations but shares feature representations. 
                  Local normalization adapts to each farm's data distribution naturally.
                </p>
              </div> */}
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">FedBN Model Structure</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="rounded-lg bg-muted p-4">
                <code className="text-xs text-muted-foreground">
                  <span className="text-primary">// Model separation</span>
                  <br />
                  <br />
                  <span className="text-accent">GLOBAL (aggregated)</span>:
                  <br />
                  &nbsp;&nbsp;Conv layers, Dense layers, Activation functions
                  <br />
                  <br />
                  <span className="text-accent">LOCAL (not aggregated)</span>:
                  <br />
                  &nbsp;&nbsp;Batch Normalization parameters
                  <br />
                  &nbsp;&nbsp;Running mean, running variance, scale, bias
                  <br />
                  <br />
                  <span className="text-primary">// During aggregation</span>
                  <br />
                  &nbsp;&nbsp;w_global ← aggregate(non_BN_weights)
                  <br />
                  &nbsp;&nbsp;BN_params_k ← kept locally
                </code>
              </div>
              <div>
                <h4 className="font-semibold text-xs text-foreground mb-2">Advantages</h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  <li>• Minimal communication overhead</li>
                  <li>• Natural personalization mechanism</li>
                  <li>• Compatible with standard networks</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-xs text-foreground mb-2">Trade-offs</h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  <li>• Requires networks with BN layers</li>
                  <li>• Less personalization than FedPer</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* FedPer Detailed Explanation */}
      <section className="bg-muted/30 mx-auto max-w-7xl px-4 py-12 sm:px-6 sm:py-16 lg:px-8">
        <div className="grid gap-8 lg:grid-cols-2 items-start">
          <div>
            <SectionHeader
              badge="Algorithm"
              title="FedPer: Federated Personalization"
              description="Explicit model separation into global and personalized components for maximum customization."
            />

            <div className="mt-6 space-y-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">Definition</h4>
                <p className="text-sm text-muted-foreground">
                  FedPer partitions each model into two distinct parts: a shared base model (aggregated globally) 
                  and personalized head model (kept locally). This enables both collaborative learning and individual customization.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">How It Works</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">1.</span> Define split point: base layers (shared) + head layers (personal)
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">2.</span> Server sends base model to all clients
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">3.</span> Each client trains full model (base + head) on local data
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">4.</span> Only base model weights sent back and aggregated
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">5.</span> Each client retains trained head model for predictions
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Why Use FedPer</h4>
                <ul className="space-y-1.5 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Strongest personalization capability
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Leverages global knowledge + local expertise
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Proven higher accuracy on heterogeneous tasks
                  </li>
                </ul>
              </div>

              {/* <div>
                <h4 className="font-semibold text-foreground mb-2">Wind Forecasting Application</h4>
                <p className="text-sm text-muted-foreground">
                  Excellent for diverse wind farms with unique local characteristics. Base model learns universal wind patterns; 
                  head learns farm-specific dynamics (terrain, obstacles, seasonal variations).
                </p>
              </div> */}
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">FedPer Architecture</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="rounded-lg bg-muted p-4">
                <code className="text-xs text-muted-foreground">
                  <span className="text-primary">// Model architecture split</span>
                  <br />
                  <br />
                  model = [base_model, head_model]
                  <br />
                  <br />
                  <span className="text-accent">// Aggregation strategy</span>
                  <br />
                  <span className="text-foreground">for</span> each round:
                  <br />
                  &nbsp;&nbsp;base_k ← train(base, head_k, local_data)
                  <br />
                  &nbsp;&nbsp;base_global ← aggregate(base_k)
                  <br />
                  &nbsp;&nbsp;head_k ← keep locally (not aggregated)
                  <br />
                  <br />
                  <span className="text-muted-foreground">// Prediction uses full model</span>
                  <br />
                  &nbsp;&nbsp;pred ← base_global + head_k
                </code>
              </div>
              <div>
                <h4 className="font-semibold text-xs text-foreground mb-2">Advantages</h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  <li>• Maximum personalization</li>
                  <li>• Learns universal + specific patterns</li>
                  <li>• Higher accuracy on diverse data</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-xs text-foreground mb-2">Trade-offs</h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  <li>• Requires careful architecture design</li>
                  <li>• More client-side storage for heads</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* SCAFFOLD Detailed Explanation */}
      <section className="mx-auto max-w-7xl px-4 py-12 sm:px-6 sm:py-16 lg:px-8">
        <div className="grid gap-8 lg:grid-cols-2 items-start">
          <div>
            <SectionHeader
              badge="Algorithm"
              title="SCAFFOLD: Controlled Variance Reduction"
              description="Uses control variates to correct client drift and achieve faster convergence."
            />

            <div className="mt-6 space-y-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">Definition</h4>
                <p className="text-sm text-muted-foreground">
                  SCAFFOLD (Stochastic Controlled Averaging For Federated Learning) addresses the problem of client drift 
                  by introducing control variates. These variates track and correct the divergence of each client's updates 
                  from the global trajectory.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">How It Works</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">1.</span> Maintain control variates: c (server) and c_k (client)
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">2.</span> Client gradient: g_k - c_k (corrected for drift)
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">3.</span> Server aggregates: Δc = c + Σ(g_k - c_k) / n
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent shrink-0">4.</span> Update control variates: c ← Δc, c_k ← updated locally
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Why Use SCAFFOLD</h4>
                <ul className="space-y-1.5 text-sm text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Reduces client drift (variance in local gradients)
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Faster convergence than FedAvg
                  </li>
                  <li className="flex gap-2">
                    <span className="text-accent">•</span> Proven theoretically for non-convex optimization
                  </li>
                </ul>
              </div>

              {/* <div>
                <h4 className="font-semibold text-foreground mb-2">Wind Forecasting Application</h4>
                <p className="text-sm text-muted-foreground">
                  Suitable when you need faster model convergence with highly distributed wind farm networks. 
                  Control variates help maintain consistent training despite regional weather variations.
                </p>
              </div> */}
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">SCAFFOLD Algorithm</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="rounded-lg bg-muted p-4">
                <code className="text-xs text-muted-foreground">
                  <span className="text-primary">// Control variate mechanism</span>
                  <br />
                  <br />
                  <span className="text-accent">// At each round:</span>
                  <br />
                  <span className="text-foreground">for</span> each client k:
                  <br />
                  &nbsp;&nbsp;Δy_k ← (g_k - c_k)  // corrected gradient
                  <br />
                  &nbsp;&nbsp;c_k ← c_k + (∇L(w_k) - g_k)  // update control
                  <br />
                  <br />
                  <span className="text-foreground">aggregate</span>:
                  <br />
                  &nbsp;&nbsp;Δc ← Σ Δy_k / n
                  <br />
                  &nbsp;&nbsp;c ← c + Δc  // server control variate
                  <br />
                  <br />
                  <span className="text-muted-foreground">// Reduces variance in gradient aggregation</span>
                </code>
              </div>
              <div>
                <h4 className="font-semibold text-xs text-foreground mb-2">Advantages</h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  <li>• Provably faster convergence</li>
                  <li>• Better variance reduction</li>
                  <li>• Works for non-convex problems</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-xs text-foreground mb-2">Trade-offs</h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  <li>• Higher communication cost (sends c_k)</li>
                  <li>• More complex implementation</li>
                  <li>• Increased client-side memory</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* LLM Models */}
      <section className="bg-muted/50 py-12 sm:py-16">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <SectionHeader
            badge="Models"
            title="Supported LLM Models for Wind Forecasting"
            description=""
            centered
          />

          <div className="mt-12 grid gap-6 md:grid-cols-2">
            {/* GPT4TS */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-primary" />
                  GPT4TS
                </CardTitle>
                <CardDescription>GPT-based Time Series Model</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                <p className="text-muted-foreground">
                  Generative Pre-trained Transformer adapted for time-series forecasting with strong performance on univariate and multivariate sequences.
                </p>
                <div className="space-y-2">
                  <h4 className="font-medium text-xs text-foreground">Key Features:</h4>
                  <ul className="space-y-1 text-xs text-muted-foreground">
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Attention mechanisms for temporal patterns
                    </li>
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Pre-trained knowledge transfer
                    </li>
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Handles irregular sampling
                    </li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            {/* LLAMA */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-primary" />
                  LLAMA
                </CardTitle>
                <CardDescription>Large Language Model Meta AI</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                <p className="text-muted-foreground">
                  Meta's open-source language model optimized for efficiency and adapted for time-series wind speed prediction.
                </p>
                <div className="space-y-2">
                  <h4 className="font-medium text-xs text-foreground">Key Features:</h4>
                  <ul className="space-y-1 text-xs text-muted-foreground">
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Efficient inference
                    </li>
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Reduced model size
                    </li>
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Better edge deployment
                    </li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            {/* BERT */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-primary" />
                  BERT
                </CardTitle>
                <CardDescription>Bidirectional Encoder Representations</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                <p className="text-muted-foreground">
                  Bidirectional transformer model that captures context from both directions for improved wind forecasting accuracy.
                </p>
                <div className="space-y-2">
                  <h4 className="font-medium text-xs text-foreground">Key Features:</h4>
                  <ul className="space-y-1 text-xs text-muted-foreground">
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Bidirectional context
                    </li>
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Strong representation learning
                    </li>
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Good for feature extraction
                    </li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            {/* BART */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-primary" />
                  BART
                </CardTitle>
                <CardDescription>Denoising Sequence-to-Sequence Model</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                <p className="text-muted-foreground">
                  Combines BERT encoder and GPT decoder for sequence-to-sequence tasks, excellent for wind speed sequence prediction.
                </p>
                <div className="space-y-2">
                  <h4 className="font-medium text-xs text-foreground">Key Features:</h4>
                  <ul className="space-y-1 text-xs text-muted-foreground">
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Sequence-to-sequence capability
                    </li>
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Denoising pre-training
                    </li>
                    <li className="flex gap-2">
                      <span className="text-accent">•</span> Multi-horizon forecasting
                    </li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

    </PageLayout>
  )
}

// Import icons that were used in the component
function Download(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
    </svg>
  )
}

function Upload(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
  )
}
