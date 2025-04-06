import { Routes } from "@angular/router";
import { BlankComponent } from "./app.component";

export const routes: Routes = [
    {
        title: "IoT Simulator",
        path: "",
        component: BlankComponent,
    },
    {
        title: "Dispositivo",
        path: ":mac",
        loadChildren: () =>
            import("./routes/panel-node/panel-node.routes").then(
                (m) => m.routes,
            ),
    },
    { path: "**", redirectTo: "" },
];
