import { registerLocaleData } from "@angular/common";
import { HttpClient, provideHttpClient } from "@angular/common/http";
import localeDe from "@angular/common/locales/de";
import localeEn from "@angular/common/locales/en";
import localeEs from "@angular/common/locales/es";
import localeFr from "@angular/common/locales/fr";
import localeIt from "@angular/common/locales/it";
import localePt from "@angular/common/locales/pt";
import {
    ApplicationConfig,
    importProvidersFrom,
    provideZoneChangeDetection,
} from "@angular/core";
import { bootstrapApplication } from "@angular/platform-browser";
import { provideAnimations } from "@angular/platform-browser/animations";
import {
    provideRouter,
    withComponentInputBinding,
    withHashLocation,
} from "@angular/router";
import { TranslateLoader, TranslateModule } from "@ngx-translate/core";
import { TranslateHttpLoader } from "@ngx-translate/http-loader";
import { ConfigService } from "@services/config.service";
import { LibraryService } from "@services/library.service";
import { ModelsService } from "@services/models.service";
import { StateService } from "@services/state.service";
import { AppComponent } from "./app/app.component";
import { routes } from "./app/app.routes";

// Registrar los idiomas para la internacionalización
registerLocaleData(localeDe, "de");
registerLocaleData(localeEn, "en");
registerLocaleData(localeEs, "es");
registerLocaleData(localeFr, "fr");
registerLocaleData(localeIt, "it");
registerLocaleData(localePt, "pt");

// Configuración de la aplicación
const appConfig: ApplicationConfig = {
    providers: [
        provideZoneChangeDetection({ eventCoalescing: true }),
        provideRouter(routes, withHashLocation(), withComponentInputBinding()),
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
