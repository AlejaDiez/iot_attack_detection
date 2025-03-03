import { ApplicationConfig, provideZoneChangeDetection } from "@angular/core";
import { bootstrapApplication } from "@angular/platform-browser";
import { provideRouter } from "@angular/router";
import { AppComponent } from "@app/app.component";
import { routes } from "@app/app.routes";

const appConfig: ApplicationConfig = {
    providers: [
        provideZoneChangeDetection({ eventCoalescing: true }),
        provideRouter(routes),
    ],
};

bootstrapApplication(AppComponent, appConfig).catch((err) =>
    console.error(err),
);
