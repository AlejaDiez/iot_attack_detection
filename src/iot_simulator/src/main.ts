import { ApplicationConfig, provideZoneChangeDetection } from "@angular/core";
import { bootstrapApplication } from "@angular/platform-browser";
import { provideAnimations } from "@angular/platform-browser/animations";
import { provideRouter, withComponentInputBinding } from "@angular/router";
import { ModelsService } from "@services/models.service";
import { AppComponent } from "./app/app.component";
import { routes } from "./app/app.routes";

const appConfig: ApplicationConfig = {
    providers: [
        provideZoneChangeDetection({ eventCoalescing: true }),
        provideRouter(routes, withComponentInputBinding()),
        provideAnimations(),
        {
            provide: ModelsService,
            useFactory: () => ModelsService.instance,
        },
    ],
};

bootstrapApplication(AppComponent, appConfig).catch((err) =>
    console.error(err),
);
