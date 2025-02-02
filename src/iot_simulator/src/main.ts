import { ApplicationConfig, provideZoneChangeDetection } from "@angular/core";
import { bootstrapApplication } from "@angular/platform-browser";
import { AppComponent } from "./app/app.component";

const appConfig: ApplicationConfig = {
    providers: [provideZoneChangeDetection({ eventCoalescing: true })],
};

bootstrapApplication(AppComponent, appConfig).catch((err) =>
    console.error(err)
);
