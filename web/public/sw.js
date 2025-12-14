const CACHE_NAME = 'typeit-v1';
const urlsToCache = [
  '/',
  '/classifier.css',
  '/latex-logo-trimmed.webp',
  '/computer-modern.otf'
];

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(cachedResponse => {
        const fetchPromise = fetch(event.request).then(fetchResponse => {
          // Cache WASM, JS, and other assets dynamically
          if (fetchResponse && fetchResponse.status === 200) {
            const responseUrl = new URL(event.request.url);
            if (responseUrl.pathname.startsWith('/pkg/') || 
                responseUrl.pathname.endsWith('.wasm') ||
                responseUrl.pathname.endsWith('.js') ||
                responseUrl.pathname.startsWith('/symbols/') ||
                responseUrl.pathname === '/' ||
                responseUrl.pathname.endsWith('.css')) {
              const responseToCache = fetchResponse.clone();
              caches.open(CACHE_NAME).then(cache => {
                cache.put(event.request, responseToCache);
              });
            }
          }
          return fetchResponse;
        }).catch(() => {
          // Network failed, return cached if available
          return cachedResponse;
        });
        
        // Return cached response immediately if available (stale-while-revalidate)
        // Otherwise wait for network
        return cachedResponse || fetchPromise;
      })
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});
