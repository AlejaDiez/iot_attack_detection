import { Routes } from "@angular/router";

export const routes: Routes = [
    {
        path: "node/:id",
        loadChildren: () =>
            import("./routes/node/node.routes").then((m) => m.routes),
    },
];
