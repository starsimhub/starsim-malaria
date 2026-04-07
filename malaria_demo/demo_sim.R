library(deSolve)
library(ggplot2)

# ============================================================
# ODE with time-varying seasonal mosquito density
# ============================================================
malaria.ode.seasonal <- function(t, x, params) {
  X <- x[1:params[["numpatch"]]]
  C <- x[(params[["numpatch"]] + 1):length(x)]

  # Seasonal mosquito density: m(t) = m_base * seasonal_factor(t)
  m <- params[["m_base"]] * params[["seasonal_fn"]](t)

  k <- as.vector(t(X * params[["H"]]) %*% params[["pij"]] /
                   (t(params[["H"]]) %*% params[["pij"]]))

  dC <- (t(m) * params[["a"]]^2 * params[["b"]] * params[["c"]] *
           exp(-params[["mu"]] * params[["tau"]]) * k /
           (params[["a"]] * params[["c"]] * k + params[["mu"]])) %*%
    t(params[["pij"]]) * (1 - X)

  dX <- dC - (params[["r"]] * X)

  return(list(c(dX, dC)))
}

# Analytical mosquito density (used for baseline m)
malaria.ode.residence.analytical <- function(Ivector, Hvector, Pmatrix, a, b, c, mu, r, tau) {
  Gvector <- function(Xvector, r) Xvector * r / (1 - Xvector)
  Fvector <- function(Xvector, Hvector, Pmatrix, a, b, c, mu) {
    k <- sapply(1:length(Xvector), function(i)
      sum(Pmatrix[, i] * Xvector * Hvector) / sum(Pmatrix[, i] * Hvector))
    return(b * c * k / (a * c * k / mu + 1))
  }
  xvector <- Ivector / r
  gvector <- Gvector(xvector, r)
  fvector <- Fvector(xvector, Hvector, Pmatrix, a, b, c, mu)
  fmatrix <- diag(fvector)
  Cvector <- solve(Pmatrix %*% fmatrix) %*% gvector
  m <- Cvector * mu / a / a / exp(-mu * tau)
  return(m)
}

# ============================================================
# Seasonal forcing functions (inspired by MicroMoB RM_mosquito)
# ============================================================

# Sinusoidal forcing on mosquito density
# amplitude: strength of seasonality (0 = constant, 1 = goes to zero at trough)
# peak_day: day of year when mosquito density peaks (default 180 = mid-year)
make_seasonal_sinusoidal <- function(amplitude = 0.8, peak_day = 180) {
  function(t) {
    day_of_year <- t %% 365
    max(0, 1 + amplitude * sin(2 * pi * (day_of_year - peak_day + 365 / 4) / 365))
  }
}

# Normal-curve forcing (like MicroMoB emergence): peaks sharply mid-year
make_seasonal_normal <- function(peak_day = 180, sd_days = 60, baseline = 0.1) {
  peak_val <- dnorm(peak_day, mean = peak_day, sd = sd_days)
  function(t) {
    day_of_year <- t %% 365
    baseline + (1 - baseline) * dnorm(day_of_year, mean = peak_day, sd = sd_days) / peak_val
  }
}

# --- Parameters ---
a <- 0.3
b <- 0.1
c <- 0.214
r <- 1 / 150
mu <- 1 / 10
tau <- 10

# --- Synthetic 3-patch data ---
numpatch <- 3
hvector <- c(5000, 10000, 8000)
ivector <- c(0.001, 0.003, 0.002)

# Mobility matrix (rows=residence, cols=destination; rows sum to 1)
pij <- matrix(c(
  0.85, 0.10, 0.05,
  0.08, 0.80, 0.12,
  0.06, 0.09, 0.85
), nrow = 3, byrow = TRUE)

# --- Solve for baseline mosquito density ---
m_base <- malaria.ode.residence.analytical(ivector, hvector, pij, a, b, c, mu, r, tau)
m_base[m_base < 0] <- 1e-10
m_base <- as.vector(m_base)
cat("Baseline mosquito density (m):", round(m_base, 4), "\n")

# --- Choose seasonal forcing ---
# Sinusoidal: smooth oscillation (change amplitude, peak_day to taste)
seasonal_fn <- make_seasonal_sinusoidal(amplitude = 0.8, peak_day = 180)
# Alternative: sharp seasonal pulse
# seasonal_fn <- make_seasonal_normal(peak_day = 180, sd_days = 60, baseline = 0.1)

# --- Set up simulation ---
nyears <- 5
times <- seq(1, 365 * nyears, by = 1)

paras <- list(numpatch = numpatch, c = c, b = b, a = a, mu = mu,
              tau = tau, r = r, pij = pij, H = hvector,
              m_base = m_base, seasonal_fn = seasonal_fn)

# Start from equilibrium prevalence (burns in faster)
xstart <- c(ivector / r, rep(0, numpatch))

cat("Running seasonal simulation for", nyears, "years...\n")
out <- as.data.frame(ode(xstart, times, malaria.ode.seasonal, paras, method = "lsoda"))
cat("Simulation complete.\n")

# ============================================================
# Derived quantities
# ============================================================
patch_names <- paste("Patch", 1:numpatch)

# Prevalence
prev_mat <- as.matrix(out[, 2:(numpatch + 1)])

# Daily incidence (diff of cumulative)
cum_mat <- as.matrix(out[, (numpatch + 2):(2 * numpatch + 1)])
daily_inc <- rbind(cum_mat[1, ], diff(cum_mat))

# Seasonal m(t) for each patch over time
m_seasonal <- t(sapply(times, function(t) m_base * seasonal_fn(t)))

# EIR: entomological inoculation rate = m(t) * a * Z
# where Z (sporozoite rate) = a*c*k*exp(-mu*tau) / (a*c*k + mu)
# Approximate k from current prevalence
eir_mat <- matrix(NA, nrow = length(times), ncol = numpatch)
for (ti in seq_along(times)) {
  X <- prev_mat[ti, ]
  k <- as.vector(t(X * hvector) %*% pij / (t(hvector) %*% pij))
  Z <- a * c * k * exp(-mu * tau) / (a * c * k + mu)
  eir_mat[ti, ] <- as.vector(m_seasonal[ti, ] * a * Z %*% t(pij))
}

# ============================================================
# PLOTS
# ============================================================
pdf("demo_sim_plots.pdf", width = 12, height = 10)

# --- 1. Seasonal forcing: mosquito density multiplier over one year ---
one_year <- 1:365
forcing_vals <- sapply(one_year, seasonal_fn)
forcing_df <- data.frame(day = one_year, multiplier = forcing_vals)
print(
  ggplot(forcing_df, aes(x = day, y = multiplier)) +
    geom_line(linewidth = 1, color = "darkgreen") +
    geom_hline(yintercept = 1, linetype = "dashed", alpha = 0.5) +
    labs(title = "Seasonal forcing: mosquito density multiplier",
         x = "Day of year", y = "Multiplier on baseline m") +
    theme_minimal()
)

# --- 2. Mosquito density m(t) by patch ---
m_df <- data.frame(time = times, m_seasonal)
names(m_df)[2:(numpatch + 1)] <- patch_names
m_long <- reshape(m_df, direction = "long", varying = patch_names,
                  v.names = "mosquito_density", timevar = "patch",
                  times = patch_names)
print(
  ggplot(m_long, aes(x = time, y = mosquito_density, color = patch)) +
    geom_line(linewidth = 0.6) +
    labs(title = "Mosquito density m(t) by patch (seasonal)",
         x = "Time (days)", y = "Mosquitoes per person (m)",
         color = "Patch") +
    theme_minimal()
)

# --- 3. Prevalence dynamics (endemic seasonal) ---
prev_df <- data.frame(time = out$time, prev_mat)
names(prev_df)[2:(numpatch + 1)] <- patch_names
prev_long <- reshape(prev_df, direction = "long", varying = patch_names,
                     v.names = "prevalence", timevar = "patch",
                     times = patch_names)
print(
  ggplot(prev_long, aes(x = time, y = prevalence, color = patch)) +
    geom_line(linewidth = 0.7) +
    labs(title = "Endemic prevalence dynamics with seasonal forcing",
         x = "Time (days)", y = "Prevalence (proportion infected)",
         color = "Patch") +
    theme_minimal()
)

# --- 4. Daily incidence by patch ---
inc_df <- data.frame(time = out$time, daily_inc)
names(inc_df)[2:(numpatch + 1)] <- patch_names
inc_long <- reshape(inc_df, direction = "long", varying = patch_names,
                    v.names = "incidence", timevar = "patch",
                    times = patch_names)
print(
  ggplot(inc_long, aes(x = time, y = incidence, color = patch)) +
    geom_line(linewidth = 0.5, alpha = 0.8) +
    labs(title = "Daily incidence (new infections per person per day)",
         x = "Time (days)", y = "Daily incidence",
         color = "Patch") +
    theme_minimal()
)

# --- 5. EIR by patch ---
eir_df <- data.frame(time = out$time, eir_mat)
names(eir_df)[2:(numpatch + 1)] <- patch_names
eir_long <- reshape(eir_df, direction = "long", varying = patch_names,
                    v.names = "EIR", timevar = "patch",
                    times = patch_names)
print(
  ggplot(eir_long, aes(x = time, y = EIR, color = patch)) +
    geom_line(linewidth = 0.6) +
    labs(title = "Entomological Inoculation Rate (EIR) by patch",
         x = "Time (days)", y = "EIR (infectious bites per person per day)",
         color = "Patch") +
    theme_minimal()
)

# --- 6. Zoomed last year: prevalence + incidence side by side ---
last_year <- out$time > 365 * (nyears - 1)

prev_zoom <- prev_long[prev_long$time > 365 * (nyears - 1), ]
print(
  ggplot(prev_zoom, aes(x = time, y = prevalence, color = patch)) +
    geom_line(linewidth = 0.8) +
    labs(title = sprintf("Prevalence: year %d (stabilized seasonal cycle)", nyears),
         x = "Time (days)", y = "Prevalence",
         color = "Patch") +
    theme_minimal()
)

inc_zoom <- inc_long[inc_long$time > 365 * (nyears - 1), ]
print(
  ggplot(inc_zoom, aes(x = time, y = incidence, color = patch)) +
    geom_line(linewidth = 0.8) +
    labs(title = sprintf("Daily incidence: year %d (stabilized seasonal cycle)", nyears),
         x = "Time (days)", y = "Daily incidence",
         color = "Patch") +
    theme_minimal()
)

dev.off()
cat("\nPlots saved to demo_sim_plots.pdf\n")
cat("Done.\n")
