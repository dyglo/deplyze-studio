import * as ort from 'onnxruntime-web';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const PROXY_URL = `${BACKEND_URL}/api/proxy/model?url=`;

const sessionCache = new Map();
const textCache = new Map();
const MODEL_CACHE_NAME = 'deplyze-browser-models-v1';

export async function getOrCreateSession(modelConfig) {
  const key = modelConfig.sessionKey ?? modelConfig.modelUrl;
  if (!key) {
    throw new Error(`No ONNX source configured for ${modelConfig.displayName}`);
  }

  if (!sessionCache.has(key)) {
    sessionCache.set(
      key,
      ort.InferenceSession.create(modelConfig.modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      }),
    );
  }

  return sessionCache.get(key);
}

export function clearSessionCache() {
  sessionCache.clear();
}

export async function ensureBrowserModelDownloaded(modelConfig) {
  let sourceUrl = modelConfig.cdnUrl ?? modelConfig.modelUrl;
  if (!sourceUrl) {
    throw new Error(`No CDN source configured for ${modelConfig.displayName}`);
  }

  // Use the backend proxy for external URLs that block CORS (like GitHub Releases)
  if (sourceUrl.includes('github.com') && sourceUrl.includes('releases/download')) {
    sourceUrl = `${PROXY_URL}${encodeURIComponent(sourceUrl)}`;
  }
  await getOrCreateSession({
    ...modelConfig,
    modelUrl: sourceUrl,
    sessionKey: modelConfig.sessionKey ?? modelConfig.id ?? sourceUrl,
  });
  return sourceUrl;
}

export async function fetchTextCached(url) {
  if (!url) {
    throw new Error('No text URL provided');
  }
  if (textCache.has(url)) {
    return textCache.get(url);
  }

  let response = null;
  if (typeof caches !== 'undefined') {
    const cache = await caches.open(MODEL_CACHE_NAME);
    const cached = await cache.match(url);
    if (cached) {
      response = cached.clone();
    } else {
      let fetchUrl = url;
      if (url.includes('github.com') || url.includes('raw.githubusercontent.com')) {
        fetchUrl = `${PROXY_URL}${encodeURIComponent(url)}`;
      }
      response = await fetch(fetchUrl, { mode: 'cors' });
      if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status}`);
      }
      await cache.put(url, response.clone());
    }
  } else {
    let fetchUrl = url;
    if (url.includes('github.com') || url.includes('raw.githubusercontent.com')) {
      fetchUrl = `${PROXY_URL}${encodeURIComponent(url)}`;
    }
    response = await fetch(fetchUrl, { mode: 'cors' });
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status}`);
    }
  }

  const text = await response.text();
  textCache.set(url, text);
  return text;
}
