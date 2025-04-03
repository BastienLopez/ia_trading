/**
 * Script principal pour l'application Crypto Trading IA
 */

document.addEventListener('DOMContentLoaded', function() {
    // Activer les tooltips Bootstrap
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Formater les nombres avec séparateurs de milliers
    function formatNumber(number, decimals = 2) {
        return number.toLocaleString('fr-FR', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }
    
    // Formater les prix en USD
    function formatPrice(price, decimals = 2) {
        return '$' + formatNumber(price, decimals);
    }
    
    // Formater les pourcentages
    function formatPercent(percent, decimals = 2) {
        return (percent >= 0 ? '+' : '') + formatNumber(percent, decimals) + '%';
    }
    
    // Formater les dates
    function formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('fr-FR') + ' ' + 
               date.toLocaleTimeString('fr-FR', {hour: '2-digit', minute: '2-digit'});
    }
    
    // Générer des couleurs aléatoires pour les graphiques
    function getRandomColor() {
        const letters = '0123456789ABCDEF';
        let color = '#';
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }
    
    // Générer un dégradé de couleur pour les graphiques
    function generateGradient(ctx, startColor, endColor) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, startColor);
        gradient.addColorStop(1, endColor);
        return gradient;
    }
    
    // Afficher un message de notification
    function showNotification(message, type = 'info') {
        // Vérifier si l'élément de notification existe
        let notificationContainer = document.getElementById('notification-container');
        
        // Créer le conteneur s'il n'existe pas
        if (!notificationContainer) {
            notificationContainer = document.createElement('div');
            notificationContainer.id = 'notification-container';
            notificationContainer.style.position = 'fixed';
            notificationContainer.style.top = '20px';
            notificationContainer.style.right = '20px';
            notificationContainer.style.zIndex = '9999';
            document.body.appendChild(notificationContainer);
        }
        
        // Créer la notification
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show`;
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Ajouter la notification au conteneur
        notificationContainer.appendChild(notification);
        
        // Supprimer automatiquement après 5 secondes
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 150);
        }, 5000);
    }
    
    // Fonction pour charger les données depuis l'API avec gestion d'erreur
    async function fetchWithErrorHandling(url, options = {}) {
        try {
            const response = await fetch(url, options);
            
            // Vérifier si la réponse est ok (status 200-299)
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Une erreur est survenue');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Erreur lors de la requête API:', error);
            showNotification(`Erreur: ${error.message}`, 'danger');
            throw error;
        }
    }
    
    // Exporter les fonctions utilitaires
    window.tradingApp = {
        formatNumber,
        formatPrice,
        formatPercent,
        formatDate,
        getRandomColor,
        generateGradient,
        showNotification,
        fetchWithErrorHandling
    };
    
    // Fonction pour créer un bouton de retour en haut de page
    function createBackToTopButton() {
        const button = document.createElement('button');
        button.id = 'back-to-top';
        button.innerHTML = '<i class="fas fa-arrow-up"></i>';
        button.className = 'btn btn-primary btn-sm rounded-circle';
        button.style.position = 'fixed';
        button.style.bottom = '20px';
        button.style.right = '20px';
        button.style.display = 'none';
        button.style.zIndex = '999';
        button.style.width = '40px';
        button.style.height = '40px';
        
        document.body.appendChild(button);
        
        // Afficher/masquer le bouton en fonction du défilement
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                button.style.display = 'block';
            } else {
                button.style.display = 'none';
            }
        });
        
        // Action de retour en haut de page
        button.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
    
    // Créer le bouton de retour en haut
    createBackToTopButton();
}); 