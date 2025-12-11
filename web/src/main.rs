use web::app::{shell, App};

#[cfg(feature = "ssr")]
#[tokio::main]
async fn main() {
    use axum::Router;
    use leptos::logging::log;
    use leptos::prelude::*;
    use leptos_axum::{generate_route_list, LeptosRoutes};

    let conf = get_configuration(Some("./leptos_options.toml")).unwrap();
    // Render sets the PORT environment variable
    let mut leptos_options = conf.leptos_options;

    // Override address to 0.0.0.0:10000 in release mode
    #[cfg(not(debug_assertions))]
    {
        leptos_options.site_addr = "0.0.0.0:10000".parse().unwrap();
    }

    log!("Site Address Loaded: {:?}", leptos_options.site_addr);
    log!("Site Root Loaded: {:?}", leptos_options.site_root);
    log!("Output Name Loaded: {:?}", leptos_options.output_name);

    let addr = leptos_options.site_addr.clone();

    // Generate the list of routes in your Leptos App
    let routes = generate_route_list(App);

    let app = Router::new()
        .leptos_routes(&leptos_options, routes, {
            let leptos_options = leptos_options.clone();
            move || shell(leptos_options.clone())
        })
        .fallback(leptos_axum::file_and_error_handler(shell))
        .with_state(leptos_options);

    // run our app with hyper
    // `axum::Server` is a re-export of `hyper::Server`
    log!("listening on http://{}", &addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

#[cfg(not(feature = "ssr"))]
pub fn main() {
    // no client-side main function
    // unless we want this to work with e.g., Trunk for pure client-side testing
    // see lib.rs for hydration function instead
}
