import { Routes } from "@angular/router";

export const routes: Routes = [
    {
        title: "Configuración",
        path: "node/:mac",
        loadChildren: () =>
            import("./routes/node/node.routes").then((m) => m.routes),
    },
];
