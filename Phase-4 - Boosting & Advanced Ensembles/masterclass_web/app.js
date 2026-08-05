// Application Logic for Classical ML Masterclass Web Dashboard (Days 050 - 056)

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSearch();
  initCodeCopy();
  initCharts();
  initXGBoostCalculator();
  initMobileToggle();
});

// Sidebar & Tab Navigation
function initNavigation() {
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('.day-section');

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const targetDay = link.getAttribute('data-day');

      // Update Nav Active State
      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');

      // Update Active Section
      sections.forEach(sec => {
        if (sec.id === targetDay) {
          sec.classList.add('active');
        } else {
          sec.classList.remove('active');
        }
      });

      // Scroll to top of content
      window.scrollTo({ top: 0, behavior: 'smooth' });

      // Close sidebar on mobile
      const sidebar = document.querySelector('.sidebar');
      if (sidebar.classList.contains('open')) {
        sidebar.classList.remove('open');
      }
    });
  });
}

// Search Filtering
function initSearch() {
  const searchInput = document.getElementById('searchInput');
  const navItems = document.querySelectorAll('.nav-item');

  if (!searchInput) return;

  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    navItems.forEach(item => {
      const text = item.textContent.toLowerCase();
      if (text.includes(query)) {
        item.style.display = 'block';
      } else {
        item.style.display = 'none';
      }
    });
  });
}

// Copy Code Button
function initCodeCopy() {
  const copyBtns = document.querySelectorAll('.btn-copy');
  copyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const codeBlock = btn.parentElement.nextElementSibling.querySelector('code');
      if (codeBlock) {
        navigator.clipboard.writeText(codeBlock.textContent).then(() => {
          const originalText = btn.textContent;
          btn.textContent = 'Copied!';
          btn.style.background = '#10b981';
          btn.style.color = '#fff';
          setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = '';
            btn.style.color = '';
          }, 2000);
        });
      }
    });
  });
}

// Chart.js Visualizations
function initCharts() {
  // Common Chart Theme Options
  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#9ca3af', font: { family: 'Inter' } } }
    },
    scales: {
      x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } },
      y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } }
    }
  };

  // Day 051 Chart: Resampling Distribution (SMOTE + Tomek Links)
  const ctx51 = document.getElementById('chartDay051');
  if (ctx51) {
    new Chart(ctx51, {
      type: 'bar',
      data: {
        labels: ['Majority Class 0', 'Minority Class 1'],
        datasets: [
          {
            label: 'Original Training Data',
            data: [1520, 80],
            backgroundColor: 'rgba(244, 63, 94, 0.6)',
            borderColor: '#f43f5e',
            borderWidth: 1
          },
          {
            label: 'After SMOTE + Tomek Links',
            data: [1485, 1485],
            backgroundColor: 'rgba(16, 185, 129, 0.6)',
            borderColor: '#10b981',
            borderWidth: 1
          }
        ]
      },
      options: chartDefaults
    });
  }

  // Day 052 Chart: Calibration Reliability Curve
  const ctx52 = document.getElementById('chartDay052');
  if (ctx52) {
    new Chart(ctx52, {
      type: 'line',
      data: {
        labels: ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
        datasets: [
          {
            label: 'Perfect Calibration',
            data: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            borderColor: '#6b7280',
            borderDash: [5, 5],
            fill: false
          },
          {
            label: 'Uncalibrated Random Forest (BS = 0.082)',
            data: [0.02, 0.08, 0.25, 0.78, 0.94, 0.98],
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            tension: 0.3
          },
          {
            label: 'Isotonic Calibrated (BS = 0.041)',
            data: [0.0, 0.19, 0.41, 0.61, 0.79, 0.99],
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            tension: 0.2
          }
        ]
      },
      options: chartDefaults
    });
  }

  // Day 053 Chart: SHAP Feature Attributions
  const ctx53 = document.getElementById('chartDay053');
  if (ctx53) {
    new Chart(ctx53, {
      type: 'bar',
      data: {
        labels: ['Age', 'Capital Gain', 'Hours per Week', 'Relationship', 'Education Num', 'Marital Status'],
        datasets: [{
          label: 'Mean |SHAP Value| (Impact on Model Output)',
          data: [1.42, 1.15, 0.88, 0.76, 0.64, 0.52],
          backgroundColor: [
            '#6366f1', '#818cf8', '#06b6d4', '#10b981', '#a855f7', '#f59e0b'
          ]
        }]
      },
      options: {
        indexAxis: 'y',
        ...chartDefaults
      }
    });
  }

  // Day 054 Chart: Anomaly Detection Scatter Simulation
  const ctx54 = document.getElementById('chartDay054');
  if (ctx54) {
    const normalPoints = Array.from({ length: 60 }, () => ({
      x: (Math.random() - 0.5) * 3,
      y: (Math.random() - 0.5) * 3
    }));
    const anomalyPoints = [
      { x: -5.2, y: 4.8 }, { x: 5.8, y: -4.2 }, { x: -4.9, y: -5.1 }, { x: 6.1, y: 5.5 }
    ];

    new Chart(ctx54, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Inliers (Normal Metrics)',
            data: normalPoints,
            backgroundColor: '#10b981'
          },
          {
            label: 'Isolated Anomalies (s -> 1.0)',
            data: anomalyPoints,
            backgroundColor: '#f43f5e',
            pointRadius: 7,
            pointHoverRadius: 9
          }
        ]
      },
      options: chartDefaults
    });
  }

  // Day 055 Chart: Classical ML Benchmark Leaderboard
  const ctx55 = document.getElementById('chartDay055');
  if (ctx55) {
    new Chart(ctx55, {
      type: 'bar',
      data: {
        labels: ['CatBoost', 'LightGBM', 'XGBoost', 'Random Forest', 'SVM (RBF)', 'Logistic Reg.'],
        datasets: [
          {
            label: 'F1-Score',
            data: [0.8654, 0.8592, 0.8541, 0.8210, 0.7845, 0.7321],
            backgroundColor: '#6366f1'
          },
          {
            label: 'ROC-AUC Score',
            data: [0.9321, 0.9288, 0.9245, 0.8990, 0.8650, 0.8112],
            backgroundColor: '#06b6d4'
          }
        ]
      },
      options: chartDefaults
    });
  }
}

// Day 056 Interactive XGBoost Calculator
function initXGBoostCalculator() {
  const gInput = document.getElementById('calcGradients');
  const hInput = document.getElementById('calcHessians');
  const lambdaInput = document.getElementById('calcLambda');

  const resW = document.getElementById('resWeight');
  const resSim = document.getElementById('resSimilarity');

  if (!gInput || !hInput || !lambdaInput || !resW || !resSim) return;

  function calculate() {
    try {
      const gVals = gInput.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
      const hVals = hInput.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
      const regLambda = parseFloat(lambdaInput.value) || 0;

      if (gVals.length === 0 || hVals.length === 0) return;

      const G = gVals.reduce((a, b) => a + b, 0);
      const H = hVals.reduce((a, b) => a + b, 0);

      const wStar = - G / (H + regLambda);
      const sim = (G * G) / (H + regLambda);

      resW.textContent = wStar.toFixed(4);
      resSim.textContent = sim.toFixed(4);
    } catch (e) {
      console.error(e);
    }
  }

  [gInput, hInput, lambdaInput].forEach(inp => {
    inp.addEventListener('input', calculate);
  });

  calculate();
}

// Mobile Sidebar Toggle
function initMobileToggle() {
  const mobileBtn = document.getElementById('mobileToggle');
  const sidebar = document.querySelector('.sidebar');
  if (mobileBtn && sidebar) {
    mobileBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }
}
