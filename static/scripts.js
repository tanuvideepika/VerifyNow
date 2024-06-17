document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById('gaugeChart').getContext('2d');
    
    const credibility_score = 75;  // Example percentage value
    
    new Chart(ctx, {
        type: 'gauge',
        data: {
            datasets: [{
                data: [credibility_score],
                value: credibility_score,
                backgroundColor: ['green', 'yellow', 'red'],
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: 'Credibility Score'
            },
            layout: {
                padding: {
                    top: 20,
                    bottom: 20
                }
            },
            needle: {
                radiusPercentage: 2,
                widthPercentage: 3.2,
                lengthPercentage: 80,
                color: 'rgba(0, 0, 0, 1)'
            },
            valueLabel: {
                formatter: Math.round,
                display: true,
                backgroundColor: '#fff',
                borderRadius: 5,
                color: '#000',
                fontSize: 24,
                fontStyle: 'normal',
                padding: 10
            }
        }
    });
});