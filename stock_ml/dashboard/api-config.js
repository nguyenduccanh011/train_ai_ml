// API Configuration for Dashboard
// Standardized to always use /api/v1

(function() {
  window.API_CONFIG = {
    baseUrl: '/api/v1',
    timeout: 10000,
    retries: 3
  };

  console.log('API Config loaded:', window.API_CONFIG);
})();
