document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('emaChart').getContext('2d');
    let emaChart = null;

    function updateEmaChart(data) {
        const config = {
            type: 'line',
            data: {
                labels: data.timestamps,
                datasets: [{
                    label: 'EMA 5',
                    data: data.ema_5,
                    borderColor: '#4CAF50',
                    tension: 0.3
                },{
                    label: 'EMA 30',
                    data: data.ema_30,
                    borderColor: '#2196F3',
                    tension: 0.3
                },{
                    label: 'EMA 50',
                    data: data.ema_50,
                    borderColor: '#FF5722',
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' },
                    tooltip: { mode: 'index' }
                },
                scales: {
                    x: { display: false },
                    y: { title: { display: true, text: 'Valeur' } }
                }
            }
        };

        if (emaChart) emaChart.destroy();
        emaChart = new Chart(ctx, config);
        
        // Mise à jour des indicateurs
        document.getElementById('ribbonWidth').textContent = 
            data.ribbon_width.slice(-1)[0].toFixed(4);
        
        const lastGradient = data.gradient.slice(-1)[0];
        document.getElementById('emaGradient').textContent = 
            (lastGradient * 100).toFixed(2) + '%';
            
        document.getElementById('emaPosition').textContent = 
            lastGradient > 0 ? '🟢 Haussière' : '🔴 Baissière';
    }

    function loadEmaData() {
        fetch('/api/ema-metrics')
            .then(r => r.json())
            .then(updateEmaChart)
            .catch(console.error);
    }

    // Chargement initial + mise à jour toutes les 30s
    loadEmaData();
    setInterval(loadEmaData, 30000);
}); 