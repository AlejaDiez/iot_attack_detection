import { ApplicationConfig, provideZoneChangeDetection } from "@angular/core";
import { bootstrapApplication } from "@angular/platform-browser";
import { provideAnimations } from "@angular/platform-browser/animations";
import { provideRouter, withComponentInputBinding } from "@angular/router";
import { LibraryService } from "@services/library.service";
import { ModelsService } from "@services/models.service";
import { StateService } from "@services/state.service";
import { AppComponent } from "./app/app.component";
import { routes } from "./app/app.routes";

const appConfig: ApplicationConfig = {
    providers: [
        provideZoneChangeDetection({ eventCoalescing: true }),
        provideRouter(routes, withComponentInputBinding()),
        provideAnimations(),
        {
            provide: StateService,
            useFactory: () => StateService.instance,
        },
        {
            provide: LibraryService,
            useFactory: () => LibraryService.instance,
        },
        {
            provide: ModelsService,
            useFactory: () => ModelsService.instance,
        },
    ],
};

bootstrapApplication(AppComponent, appConfig).catch((err) =>
    console.error(err),
);
