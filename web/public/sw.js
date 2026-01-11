// IMPORTANT: Increment this version number when you push updates
const CACHE_VERSION = 'v3';
const CACHE_NAME = `typeit-${CACHE_VERSION}`;
const urlsToCache = [
  '/',
  '/classifier.css',
  '/latex-logo-trimmed.webp',
  '/latex-logo-trimmed-filled-in.webp',
  '/computer-modern.otf'
];

self.addEventListener('install', event => {
  // Force the waiting service worker to become the active service worker
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
          // Delete old caches
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      // Take control of all clients immediately
      return self.clients.claim();
    }).then(() => {
      // Notify all clients that an update is available
      return self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({
            type: 'SW_UPDATED',
            version: CACHE_VERSION
          });
        });
      });
    })
  );
});
