// Dynamic API Configuration for Dashboard
// Auto-detects environment (local development vs containerized production)

(function() {
  // Determine API base URL
  const determineApiBase = () => {
    // In production (Docker/Nginx): use relative path
    if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      return '/api/v1';
    }

    // In development: try to detect if running through Nginx or directly
    if (window.location.port === '80' || window.location.port === '443' || window.location.port === '') {
      // Likely behind Nginx
      return '/api/v1';
    }

    // Direct access to dashboard server
    return 'http://localhost:8000/api/v1';
  };

  // Expose global config
  window.API_CONFIG = {
    baseUrl: determineApiBase(),
    timeout: 10000,
    retries: 3
  };

  console.log('API Config loaded:', window.API_CONFIG);
})();
