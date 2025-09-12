document.addEventListener('DOMContentLoaded', () => {
    const auditButton = document.getElementById('auditButton');
    const urlInput = document.getElementById('urlInput');
    const resultsSection = document.getElementById('results');
    const loader = document.getElementById('loader');
    const errorDiv = document.getElementById('error');

    const scoreEl = document.getElementById('score');
    const keyFindingsEl = document.getElementById('keyFindings');
    const recommendationsEl = document.getElementById('recommendations');

    auditButton.addEventListener('click', async () => {
        const url = urlInput.value.trim();
        if (!url) {
            showError("Please enter a valid URL.");
            return;
        }

        // Reset UI
        resultsSection.classList.add('hidden');
        errorDiv.classList.add('hidden');
        loader.classList.remove('hidden');

        try {
            const response = await fetch('http://127.0.0.1:5001/audit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Something went wrong on the server.');
            }

            const data = await response.json();
            displayResults(data);

        } catch (err) {
            showError(err.message);
        } finally {
            loader.classList.add('hidden');
        }
    });

    function displayResults(data) {
        scoreEl.textContent = data.overall_score;
        
        // Update score circle color based on score
        const scoreCircle = scoreEl.parentElement;
        if (data.overall_score >= 75) {
            scoreCircle.style.borderColor = '#28a745'; // Green
            scoreCircle.style.backgroundColor = '#d4edda';
            scoreCircle.style.color = '#155724';
        } else if (data.overall_score >= 40) {
            scoreCircle.style.borderColor = '#ffc107'; // Yellow
            scoreCircle.style.backgroundColor = '#fff3cd';
            scoreCircle.style.color = '#856404';
        } else {
            scoreCircle.style.borderColor = '#dc3545'; // Red
            scoreCircle.style.backgroundColor = '#f8d7da';
            scoreCircle.style.color = '#721c24';
        }


        keyFindingsEl.innerHTML = '';
        data.key_findings.forEach(finding => {
            const li = document.createElement('li');
            li.textContent = finding;
            keyFindingsEl.appendChild(li);
        });

        recommendationsEl.innerHTML = '';
        data.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsEl.appendChild(li);
        });

        resultsSection.classList.remove('hidden');
    }

    function showError(message) {
        errorDiv.textContent = `Error: ${message}`;
        errorDiv.classList.remove('hidden');
    }
});