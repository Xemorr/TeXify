use leptos::prelude::*;
use leptos::html::InnerHtmlAttribute;

#[component]
pub fn InstallButton() -> impl IntoView {
    view! {
        // Install prompt
        <div id="install-container" style="display: none; text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin-bottom: 20px; border-radius: 8px;">
            <p style="color: white; margin: 5px 0; font-size: 0.9em;">
                "Install this app for quick access and offline use!"
            </p>
            <button
                id="install-button"
                style="
                    background: white;
                    color: #667eea;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-weight: bold;
                    cursor: pointer;
                    font-size: 1em;
                    transition: transform 0.2s;
                "
                class="install-button"
            >
                "📲 Install App"
            </button>
        </div>

        // Update available prompt
        <div id="update-container" style="display: none; text-align: center; padding: 10px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); margin-bottom: 20px; border-radius: 8px;">
            <p style="color: white; margin: 5px 0; font-size: 0.9em;">
                "🎉 A new version is available!"
            </p>
            <button
                id="refresh-button"
                style="
                    background: white;
                    color: #f5576c;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-weight: bold;
                    cursor: pointer;
                    font-size: 1em;
                    transition: transform 0.2s;
                "
                class="install-button"
            >
                "🔄 Refresh to Update"
            </button>
        </div>

        <script inner_html=r#"
            let deferredPrompt;

            // Install prompt handling
            window.addEventListener('beforeinstallprompt', (e) => {
                e.preventDefault();
                deferredPrompt = e;
                const container = document.getElementById('install-container');
                if (container) {
                    container.style.display = 'block';
                }
            });

            const installBtn = document.getElementById('install-button');
            if (installBtn) {
                installBtn.addEventListener('click', async () => {
                    if (!deferredPrompt) return;

                    deferredPrompt.prompt();
                    const { outcome } = await deferredPrompt.userChoice;

                    deferredPrompt = null;
                    const container = document.getElementById('install-container');
                    if (container) {
                        container.style.display = 'none';
                    }
                });
            }

            // Update notification handling
            if ('serviceWorker' in navigator) {
                navigator.serviceWorker.addEventListener('message', (event) => {
                    if (event.data.type === 'SW_UPDATED') {
                        console.log('New version available:', event.data.version);
                        const updateContainer = document.getElementById('update-container');
                        if (updateContainer) {
                            updateContainer.style.display = 'block';
                        }
                    }
                });
            }

            const refreshBtn = document.getElementById('refresh-button');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => {
                    window.location.reload();
                });
            }
        "#></script>
    }
}
