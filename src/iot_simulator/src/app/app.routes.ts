import { Routes } from "@angular/router";

export const routes: Routes = [
    {
        title: "Configuración",
        path: "node/:id",
        loadChildren: () =>
            import("./routes/node/node.routes").then((m) => m.routes),
    },
];
