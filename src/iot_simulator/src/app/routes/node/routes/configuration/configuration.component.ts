import { CommonModule } from "@angular/common";
import { Component, OnInit } from "@angular/core";
import { FormControl, FormGroup, ReactiveFormsModule } from "@angular/forms";
import { ActivatedRoute } from "@angular/router";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmInputModule } from "@spartan-ng/ui-input-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmSwitchModule } from "@spartan-ng/ui-switch-helm";
import { Device, DeviceType } from "../../../../models/device";
import { Router } from "../../../../models/router";

@Component({
    imports: [
        CommonModule,
        ReactiveFormsModule,
        BrnSelectModule,
        HlmInputModule,
        HlmLabelModule,
        HlmSelectModule,
        HlmSwitchModule,
    ],
    templateUrl: "configuration.component.html",
})
export class ConfigurationComponent implements OnInit {
    protected readonly types: string[] = Object.values(DeviceType);
    protected readonly detection_models: string[] = [
        "tensorflow",
        "federated learning",
    ];
    protected node!: Device;
    protected form: FormGroup = new FormGroup({
        name: new FormControl(""),
        ip: new FormControl(""),
        type: new FormControl(""),
        detection: new FormControl(true),
        detection_model: new FormControl("tensorflow"),
    });

    public constructor(private _route: ActivatedRoute) {}

    public ngOnInit(): void {
        this._route.parent!.params.subscribe(({ id }) => {
            if (id === "192.168.0.1") {
                this.node = new Router(id, "Router");
            } else {
                this.node = new Device(id, "Dispositivo");
            }

            this.form.get("name")!.setValue(this.node.name);
            this.form.get("ip")!.setValue(this.node.ip);
            this.form.get("type")!.setValue(this.node.type.valueOf());
        });
    }
}
