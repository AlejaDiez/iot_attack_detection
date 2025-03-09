import { Routes } from "@angular/router";

export const routes: Routes = [
    {
        title: "Configuración",
        path: ":mac",
        loadChildren: () =>
            import("./routes/node/node.routes").then((m) => m.routes),
    },
    { path: "**", redirectTo: "" },
];
