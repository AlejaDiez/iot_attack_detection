import { HttpClient, provideHttpClient } from "@angular/common/http";
import {
    ApplicationConfig,
    importProvidersFrom,
    provideZoneChangeDetection,
} from "@angular/core";
import { bootstrapApplication } from "@angular/platform-browser";
import { provideAnimations } from "@angular/platform-browser/animations";
import { provideRouter, withComponentInputBinding } from "@angular/router";
import { TranslateLoader, TranslateModule } from "@ngx-translate/core";
import { TranslateHttpLoader } from "@ngx-translate/http-loader";
import { ConfigService } from "@services/config.service";
import { LibraryService } from "@services/library.service";
import { ModelsService } from "@services/models.service";
import { StateService } from "@services/state.service";
import { AppComponent } from "./app/app.component";
import { routes } from "./app/app.routes";

const appConfig: ApplicationConfig = {
    providers: [
        provideZoneChangeDetection({ eventCoalescing: true }),
        provideRouter(routes, withComponentInputBinding()),
        provideHttpClient(),
        provideAnimations(),
        importProvidersFrom(
            TranslateModule.forRoot({
                loader: {
                    provide: TranslateLoader,
                    useFactory: (http: HttpClient) =>
                        new TranslateHttpLoader(
                            http,
                            "./assets/i18n/",
                            ".json",
                        ),
                    deps: [HttpClient],
                },
            }),
        ),
        {
            provide: StateService,
            useFactory: () => StateService.init(),
        },
        {
            provide: LibraryService,
            useFactory: (config: ConfigService) => LibraryService.init(config),
            deps: [ConfigService],
        },
        {
            provide: ModelsService,
            useFactory: (config: ConfigService) => ModelsService.init(config),
            deps: [ConfigService],
        },
    ],
};

bootstrapApplication(AppComponent, appConfig).catch((err) =>
    console.error(err),
);
